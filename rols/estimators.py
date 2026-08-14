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
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided


@dataclass(frozen=True)
class JointFitResult:
    """Outputs from one rolling joint model per endpoint and target."""

    coef: np.ndarray
    intercept: np.ndarray
    resid_endpoint: np.ndarray
    ssr: np.ndarray
    sst: np.ndarray
    n_used: np.ndarray
    n_eff: np.ndarray
    n_singular: int = 0
    n_ill_conditioned: int = 0


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


def _warn_ill_conditioned(n: int, threshold: float) -> None:
    """Emit one warning for all full-rank windows above the condition threshold."""
    if n <= 0:
        return
    warnings.warn(
        f"{n} window(s) have an ill-conditioned design "
        f"(cond(X'X) > {threshold:.0e}). Estimates in those windows may be "
        "numerically unreliable. Consider Ridge (lambda_ > 0) or removing "
        "collinear regressors.",
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
    shape = (n, window, d)
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


def _solve_joint_window(
    target: np.ndarray,
    design: np.ndarray,
    complete_case: np.ndarray,
    fit_intercept: bool,
    weights: np.ndarray | None,
    penalty: np.ndarray,
    cond_warn_threshold: float,
) -> tuple[tuple[np.ndarray, float, float, float, int, float] | None, bool]:
    """Solve one complete-case window for one target."""
    n_used = int(complete_case.sum())
    if n_used == 0:
        return None, False

    complete_target = target[complete_case]
    complete_design = design[complete_case]
    if weights is None:
        complete_weights = np.full(n_used, 1.0 / n_used)
    else:
        complete_weights = weights[complete_case]
        weight_sum = complete_weights.sum()
        if weight_sum <= 0:
            return None, False
        complete_weights = complete_weights / weight_sum

    penalty_diagonal = np.diag(penalty)
    penalized_columns = np.flatnonzero(penalty_diagonal > 0)
    solve_design = complete_design.copy()
    means = np.zeros(solve_design.shape[1])
    scales = np.ones(solve_design.shape[1])
    for column in penalized_columns:
        values = solve_design[:, column]
        if fit_intercept:
            means[column] = np.sum(complete_weights * values)
            solve_design[:, column] = values - means[column]
            scale_squared = np.sum(complete_weights * solve_design[:, column] ** 2)
        else:
            scale_squared = np.sum(complete_weights * values**2)
        scales[column] = np.sqrt(scale_squared)
        if not np.isfinite(scales[column]) or scales[column] <= np.finfo(float).eps:
            return None, False
        solve_design[:, column] /= scales[column]

    sqrt_weights = np.sqrt(complete_weights)
    weighted_design = solve_design * sqrt_weights[:, None]
    weighted_target = complete_target * sqrt_weights
    design_gram = weighted_design.T @ weighted_design
    condition_number = float(np.linalg.cond(design_gram))
    if np.any(penalty):
        augmented_design = np.vstack([weighted_design, np.diag(np.sqrt(np.diag(penalty)))])
        augmented_target = np.concatenate([weighted_target, np.zeros(solve_design.shape[1])])
    else:
        augmented_design = weighted_design
        augmented_target = weighted_target

    if np.linalg.matrix_rank(augmented_design) < solve_design.shape[1]:
        return None, False
    try:
        q, r = np.linalg.qr(augmented_design, mode="reduced")
        solve_parameters = np.linalg.solve(r, q.T @ augmented_target)
    except np.linalg.LinAlgError:
        return None, False
    if not np.isfinite(solve_parameters).all():
        return None, False

    parameters = solve_parameters / scales
    if fit_intercept:
        parameters[0] -= np.sum(parameters[1:] * means[1:])

    residuals = complete_target - complete_design @ parameters
    ssr = float(np.sum(complete_weights * residuals**2))
    if fit_intercept:
        target_mean = np.sum(complete_weights * complete_target)
        sst = float(np.sum(complete_weights * (complete_target - target_mean) ** 2))
    else:
        sst = float(np.sum(complete_weights * complete_target**2))
    endpoint_residual = np.nan
    if complete_case[-1]:
        endpoint_residual = float(target[-1] - design[-1] @ parameters)
    n_eff = float(1.0 / np.sum(complete_weights**2))
    result = parameters, endpoint_residual, ssr, sst, n_used, n_eff
    return result, condition_number > cond_warn_threshold


def rolling_joint_solve(
    y: pd.DataFrame,
    X: pd.DataFrame,
    window: int,
    min_periods: int,
    expanding: bool,
    fit_intercept: bool = True,
    penalty: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    warn_singular: bool = True,
    cond_warn_threshold: float = 1e10,
) -> JointFitResult:
    """Fit one current-window joint model per endpoint and target.

    ``X`` contains slopes only. When requested, the intercept is an explicit
    design column. Every coefficient, endpoint residual, sum of squares, and
    sample count comes from the same complete-case fit. Equal and supplied
    observation weights are normalized to sum to one within each target's
    complete-case sample. Solves use QR on the weighted design, augmented by
    the square root of the penalty for Ridge. Conditioning diagnostics use
    ``cond(X'X)`` for the weighted design before augmentation.
    """
    assert y.index.equals(X.index), "y and X must have identical indexes"
    if window <= 0 or min_periods <= 0:
        raise ValueError("window and min_periods must be positive")
    if not expanding and min_periods > window:
        raise ValueError("min_periods cannot exceed window")
    if weights is not None and expanding:
        raise ValueError("weights are not supported with expanding=True")
    if not np.isfinite(cond_warn_threshold) or cond_warn_threshold <= 0:
        raise ValueError("cond_warn_threshold must be finite and positive")

    target_values = y.to_numpy(dtype=np.float64)
    regressor_values = X.to_numpy(dtype=np.float64)
    n_observations, n_targets = target_values.shape
    n_slopes = regressor_values.shape[1]
    design = (
        np.column_stack([np.ones(n_observations), regressor_values])
        if fit_intercept
        else regressor_values
    )
    n_parameters = design.shape[1]

    if penalty is None:
        penalty_matrix = np.zeros((n_parameters, n_parameters), dtype=np.float64)
    else:
        penalty_matrix = np.asarray(penalty, dtype=np.float64)
        if penalty_matrix.shape != (n_parameters, n_parameters):
            raise ValueError(f"penalty must have shape ({n_parameters}, {n_parameters})")
        if not np.isfinite(penalty_matrix).all():
            raise ValueError("penalty must be finite")
        if not np.allclose(penalty_matrix, np.diag(np.diag(penalty_matrix))):
            raise ValueError("penalty must be diagonal")
        if (np.diag(penalty_matrix) < 0).any():
            raise ValueError("penalty entries must be non-negative")
        if fit_intercept and not np.allclose(penalty_matrix[0], 0.0):
            raise ValueError("the intercept cannot be penalized")
        if fit_intercept and not np.allclose(penalty_matrix[:, 0], 0.0):
            raise ValueError("the intercept cannot be penalized")

    supplied_weights = None
    if weights is not None:
        supplied_weights = np.asarray(weights, dtype=np.float64)
        if supplied_weights.shape != (window,):
            raise ValueError(f"weights must have shape ({window},)")
        if not np.isfinite(supplied_weights).all() or (supplied_weights < 0).any():
            raise ValueError("weights must be finite and non-negative")
        if supplied_weights.sum() <= 0:
            raise ValueError("weights must have positive sum")

    coef = np.full((n_observations, n_slopes, n_targets), np.nan)
    intercept = np.full((n_observations, n_targets), np.nan)
    resid_endpoint = np.full((n_observations, n_targets), np.nan)
    ssr = np.full((n_observations, n_targets), np.nan)
    sst = np.full((n_observations, n_targets), np.nan)
    n_used = np.full((n_observations, n_targets), np.nan)
    n_eff = np.full((n_observations, n_targets), np.nan)

    def store(
        endpoint: int,
        target_position: int,
        solved: tuple[np.ndarray, float, float, float, int, float],
    ) -> None:
        parameters, residual, window_ssr, window_sst, window_n_used, window_n_eff = solved
        if fit_intercept:
            intercept[endpoint, target_position] = parameters[0]
            coef[endpoint, :, target_position] = parameters[1:]
        else:
            intercept[endpoint, target_position] = 0.0
            coef[endpoint, :, target_position] = parameters
        resid_endpoint[endpoint, target_position] = residual
        ssr[endpoint, target_position] = window_ssr
        sst[endpoint, target_position] = window_sst
        n_used[endpoint, target_position] = window_n_used
        n_eff[endpoint, target_position] = window_n_eff

    vectorizable_rolling = not expanding and np.isfinite(design).all() and n_observations >= window
    n_singular = 0
    n_ill_conditioned = 0
    if vectorizable_rolling:
        clean_target_positions = np.flatnonzero(np.isfinite(target_values).all(axis=0))
        dirty_target_positions = np.flatnonzero(~np.isfinite(target_values).all(axis=0))
        design_windows = _make_windows(design, window)
        target_windows = _make_windows(target_values[:, clean_target_positions], window)
        window_weights = (
            np.full(window, 1.0 / window)
            if supplied_weights is None
            else supplied_weights / supplied_weights.sum()
        )
        penalty_diagonal = np.diag(penalty_matrix)
        penalized_columns = np.flatnonzero(penalty_diagonal > 0)
        solve_design_windows = design_windows.copy()
        means = np.zeros((solve_design_windows.shape[0], n_parameters))
        scales = np.ones((solve_design_windows.shape[0], n_parameters))
        for column in penalized_columns:
            values = solve_design_windows[:, :, column]
            if fit_intercept:
                means[:, column] = np.einsum("w,tw->t", window_weights, values)
                solve_design_windows[:, :, column] = values - means[:, None, column]
                scale_squared = np.einsum(
                    "w,tw->t", window_weights, solve_design_windows[:, :, column] ** 2
                )
            else:
                scale_squared = np.einsum("w,tw->t", window_weights, values**2)
            scales[:, column] = np.sqrt(scale_squared)

        valid_scales = np.isfinite(scales).all(axis=1) & (
            scales[:, penalized_columns] > np.finfo(float).eps
        ).all(axis=1)
        safe_scales = np.where(valid_scales[:, None], scales, 1.0)
        solve_design_windows /= safe_scales[:, None, :]
        sqrt_window_weights = np.sqrt(window_weights)
        weighted_design_windows = solve_design_windows * sqrt_window_weights[None, :, None]
        weighted_target_windows = target_windows * sqrt_window_weights[None, :, None]
        if np.any(penalty_matrix):
            square_root_penalty = np.diag(np.sqrt(np.diag(penalty_matrix)))
            augmented_design = np.concatenate(
                [
                    weighted_design_windows,
                    np.broadcast_to(
                        square_root_penalty,
                        (weighted_design_windows.shape[0], n_parameters, n_parameters),
                    ),
                ],
                axis=1,
            )
            augmented_targets = np.concatenate(
                [
                    weighted_target_windows,
                    np.zeros((target_windows.shape[0], n_parameters, target_windows.shape[2])),
                ],
                axis=1,
            )
        else:
            augmented_design = weighted_design_windows
            augmented_targets = weighted_target_windows

        design_gram = np.einsum("twi,twj->tij", weighted_design_windows, weighted_design_windows)
        condition_numbers = np.linalg.cond(design_gram)
        ranks = np.linalg.matrix_rank(augmented_design)
        full_rank = ranks == n_parameters
        valid_windows = valid_scales & full_rank
        if clean_target_positions.size:
            n_singular += int((~valid_windows).sum())
            n_ill_conditioned += int(
                (valid_windows & (condition_numbers > cond_warn_threshold)).sum()
            )

        q, r = np.linalg.qr(augmented_design, mode="reduced")
        projected_targets = np.einsum("twk,twn->tkn", q, augmented_targets)
        safe_r = r.copy()
        safe_r[~valid_windows] = np.eye(n_parameters)
        solve_parameters = np.linalg.solve(safe_r, projected_targets)
        solve_parameters[~valid_windows] = np.nan
        parameters = solve_parameters / safe_scales[:, :, None]
        if fit_intercept:
            parameters[:, 0, :] -= np.einsum("tk,tkn->tn", means[:, 1:], parameters[:, 1:, :])
        parameters[~valid_scales] = np.nan
        endpoints = np.arange(parameters.shape[0]) + window - 1
        fitted_windows = np.einsum("twk,tkn->twn", design_windows, parameters)
        residual_windows = target_windows - fitted_windows
        window_ssr = np.einsum("w,twn->tn", window_weights, residual_windows**2)
        if fit_intercept:
            target_means = np.einsum("w,twn->tn", window_weights, target_windows)
            centered_targets = target_windows - target_means[:, None, :]
            window_sst = np.einsum("w,twn->tn", window_weights, centered_targets**2)
        else:
            window_sst = np.einsum("w,twn->tn", window_weights, target_windows**2)
        for local_position, target_position in enumerate(clean_target_positions):
            if fit_intercept:
                intercept[endpoints, target_position] = parameters[:, 0, local_position]
                coef[endpoints, :, target_position] = parameters[:, 1:, local_position]
            else:
                intercept[endpoints, target_position] = 0.0
                coef[endpoints, :, target_position] = parameters[:, :, local_position]
            resid_endpoint[endpoints, target_position] = residual_windows[:, -1, local_position]
            ssr[endpoints, target_position] = window_ssr[:, local_position]
            sst[endpoints, target_position] = window_sst[:, local_position]
            n_used[endpoints, target_position] = window
            n_eff[endpoints, target_position] = 1.0 / np.sum(window_weights**2)

        early_stop = min(window - 1, n_observations)
        for endpoint in range(min_periods - 1, early_stop):
            endpoint_weights = (
                None if supplied_weights is None else supplied_weights[-(endpoint + 1) :]
            )
            for target_position in range(n_targets):
                complete_case = np.isfinite(target_values[: endpoint + 1, target_position])
                if int(complete_case.sum()) < min_periods:
                    continue
                solved, ill_conditioned = _solve_joint_window(
                    target_values[: endpoint + 1, target_position],
                    design[: endpoint + 1],
                    complete_case,
                    fit_intercept,
                    endpoint_weights,
                    penalty_matrix,
                    cond_warn_threshold,
                )
                n_ill_conditioned += int(ill_conditioned)
                if solved is None:
                    n_singular += 1
                else:
                    store(endpoint, target_position, solved)
        for endpoint in range(window - 1, n_observations):
            start = endpoint - window + 1
            endpoint_weights = supplied_weights
            window_design = design[start : endpoint + 1]
            for target_position in dirty_target_positions:
                window_target = target_values[start : endpoint + 1, target_position]
                complete_case = np.isfinite(window_target)
                if int(complete_case.sum()) < min_periods:
                    continue
                solved, ill_conditioned = _solve_joint_window(
                    window_target,
                    window_design,
                    complete_case,
                    fit_intercept,
                    endpoint_weights,
                    penalty_matrix,
                    cond_warn_threshold,
                )
                n_ill_conditioned += int(ill_conditioned)
                if solved is None:
                    n_singular += 1
                else:
                    store(endpoint, target_position, solved)
    else:
        for endpoint in range(min_periods - 1, n_observations):
            start = 0 if expanding else max(0, endpoint - window + 1)
            endpoint_weights = (
                None if supplied_weights is None else supplied_weights[-(endpoint - start + 1) :]
            )
            for target_position in range(n_targets):
                window_target = target_values[start : endpoint + 1, target_position]
                window_design = design[start : endpoint + 1]
                complete_case = np.isfinite(window_target) & np.isfinite(window_design).all(axis=1)
                if int(complete_case.sum()) < min_periods:
                    continue
                solved, ill_conditioned = _solve_joint_window(
                    window_target,
                    window_design,
                    complete_case,
                    fit_intercept,
                    endpoint_weights,
                    penalty_matrix,
                    cond_warn_threshold,
                )
                n_ill_conditioned += int(ill_conditioned)
                if solved is None:
                    n_singular += 1
                else:
                    store(endpoint, target_position, solved)

    if warn_singular:
        _warn_singular(n_singular)
        _warn_ill_conditioned(n_ill_conditioned, cond_warn_threshold)
    return JointFitResult(
        coef=coef,
        intercept=intercept,
        resid_endpoint=resid_endpoint,
        ssr=ssr,
        sst=sst,
        n_used=n_used,
        n_eff=n_eff,
        n_singular=n_singular,
        n_ill_conditioned=n_ill_conditioned,
    )


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
            y_w = y_col[: t + 1]
            row_ok = x_row_valid[: t + 1] & ~np.isnan(y_w)
            if row_ok.sum() < min_periods:
                continue
            Xw_c = X_np[: t + 1][row_ok]
            yw_c = y_w[row_ok]
            if weights is not None:
                w_c = weights[-(t + 1) :][row_ok]
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
    assert y.index.equals(X.index), "y and X must have identical indexes"
    if weights is not None and expanding:
        raise ValueError("weights are not supported with expanding=True")
    y_np = y.to_numpy(dtype=np.float64)
    X_np = X.to_numpy(dtype=np.float64)
    T, N = y_np.shape
    k = X_np.shape[1]
    resid = np.full((T, N), np.nan)
    ridge_term = ridge_lambda * np.eye(k)
    n_singular = 0

    if expanding:
        # Expanding window — loop required regardless (variable size)
        # Per-column NaN handling: drop rows with NaN in X or y_j
        x_row_valid = ~np.isnan(X_np).any(axis=1)  # (T,)
        for t in range(min_periods - 1, T):
            X_end = X_np[: t + 1]
            y_end = y_np[: t + 1]
            x_ok = x_row_valid[: t + 1]

            for j in range(N):
                if np.isnan(y_np[t, j]):
                    continue
                row_ok = x_ok & ~np.isnan(y_end[:, j])
                if row_ok.sum() < min_periods:
                    continue
                Xw_c = X_end[row_ok]
                yw_c = y_end[row_ok, j]
                XtX = Xw_c.T @ Xw_c + ridge_term
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
        valid = ~has_nan_X

        if weights is not None:
            # Weighted gram matrix: X'WX and X'Wy. Apply weights to one side
            # of the einsum so the accumulation sums w_t * x_t * (.)_t.
            Xw_w = Xw * weights[None, :, None]  # (n, window, k)
            XtX = np.einsum("twi,twj->tij", Xw_w, Xw)
            XtY = np.einsum("twi,twn->tin", Xw_w, yw)
        else:
            XtX = np.einsum("twi,twj->tij", Xw, Xw)
            XtY = np.einsum("twi,twn->tin", Xw, yw)
        XtX[valid] += ridge_term

        betas = np.full((n_windows, k, N), np.nan)
        if valid.any():
            # _solve_batch emits its own aggregated warning for the batch.
            betas[valid] = _solve_batch(XtX[valid], XtY[valid], warn_singular=warn_singular)

        t_idx = np.arange(n_windows) + window - 1
        fitted = np.einsum("ti,tin->tn", X_np[t_idx], betas)
        resid[t_idx] = np.where(has_nan_X[:, None], np.nan, y_np[t_idx] - fitted)

        if min_periods < window:
            for t in range(min_periods - 1, window - 1):
                Xw_t, yw_t = X_np[: t + 1], y_np[: t + 1]
                if np.isnan(Xw_t).any():
                    continue
                if weights is not None:
                    w_t = weights[-(t + 1) :]
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
            yw_j = yw[:, :, j]  # (n_windows, window)
            valid_j = ~np.isnan(yw_j)  # (n_windows, window)
            valid_count = valid_j.sum(axis=1)  # (n_windows,)
            # The prediction point is the last row of each window; if y_j is
            # NaN there the window is skipped (matching _residualize_single).
            pred_valid = valid_j[:, -1]
            sufficient = (valid_count >= min_periods) & pred_valid
            if not sufficient.any():
                continue

            # Zero out the X (and y) rows where y_j is NaN so they drop out of
            # both X'X and X'y for this asset.
            Xw_masked = np.where(valid_j[:, :, None], Xw, 0.0)  # (n_windows, window, k)
            yw_masked = np.where(valid_j, yw_j, 0.0)  # (n_windows, window)

            if weights is not None:
                # Per-window weights restricted to the surviving (non-NaN) rows,
                # renormalized to sum to 1. Insufficient windows have zero sum
                # but are filtered out by `sufficient`, so the divide is guarded.
                wm = np.where(valid_j, weights[None, :], 0.0)  # (n_windows, window)
                wsum = wm.sum(axis=1, keepdims=True)  # (n_windows, 1)
                wn = np.divide(wm, wsum, out=np.zeros_like(wm), where=wsum > 0)
                Xw_w = Xw_masked * wn[:, :, None]  # (n_windows, window, k)
                XtX_j = np.einsum("twi,twj->tij", Xw_w, Xw_masked)  # (n_windows, k, k)
                XtY_j = np.einsum("twi,tw->ti", Xw_w, yw_masked)[:, :, None]
            else:
                XtX_j = np.einsum("twi,twj->tij", Xw_masked, Xw_masked)  # (n_windows, k, k)
                XtY_j = np.einsum("twi,tw->ti", Xw_masked, yw_masked)[:, :, None]
            XtX_j[sufficient] += ridge_term

            betas_j = np.full((n_windows, k, 1), np.nan)
            # Aggregate the singular warning once for the whole call rather than
            # emitting one per asset.
            betas_j[sufficient] = _solve_batch(
                XtX_j[sufficient], XtY_j[sufficient], warn_singular=False
            )
            n_singular += int((sufficient & np.isnan(betas_j[:, :, 0]).any(axis=1)).sum())

            fitted = np.einsum("ti,ti->t", X_np[t_idx], betas_j[:, :, 0])
            resid[t_idx, j] = np.where(sufficient, y_np[t_idx, j] - fitted, np.nan)

        # Early windows (min_periods < window) use variable-size expanding
        # windows — handled per asset exactly as in _residualize_single.
        if min_periods < window:
            for t in range(min_periods - 1, window - 1):
                for j in range(N):
                    if np.isnan(y_np[t, j]):
                        continue
                    y_w = y_np[: t + 1, j]
                    row_ok = ~np.isnan(y_w)
                    if row_ok.sum() < min_periods:
                        continue
                    Xw_c = X_np[: t + 1][row_ok]
                    yw_c = y_w[row_ok]
                    if weights is not None:
                        w_c = weights[-(t + 1) :][row_ok]
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
        y_col = result[[cols[j]]]
        Xprev = result[cols[:j]]

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
    f_np = factor_values.to_numpy(dtype=np.float64)
    T, N = resid_np.shape
    se = np.full((T, N), np.nan)

    def _nw_se_window(f_w: np.ndarray, e_w: np.ndarray) -> np.ndarray:
        n_obs = len(f_w)
        score = f_w[:, None] * e_w
        xx = f_w @ f_w
        S = np.einsum("ti,ti->i", score, score) / n_obs
        for lag in range(1, n_lags + 1):
            w = 1.0 - lag / (n_lags + 1)
            gamma = np.einsum("ti,ti->i", score[lag:], score[:-lag]) / n_obs
            S += 2 * w * gamma
        var_beta = S * n_obs / (xx**2)
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
        # Expanding window — loop required (variable window size per t).
        for t in range(min_periods - 1, T):
            _fill_window(t, f_np[: t + 1], resid_np[: t + 1])
    else:
        # Rolling window — fully vectorized over T via stride tricks. The
        # Python loop is O(n_lags) (typically 3-10), with T collapsed into the
        # einsum reductions. This produces results identical to the per-window
        # loop (see test_vectorized_matches_loop) at a large speedup: at
        # T=1500, N=2300 the loop made ~1500 Python calls each doing O(n_lags*N)
        # work, whereas this makes O(n_lags) numpy calls over the whole panel.
        n_windows = T - window + 1
        if n_windows > 0 and window > n_lags:
            # (n_windows, window) and (n_windows, window, N) zero-copy views.
            f_wins = _make_windows(f_np[:, None], window)[:, :, 0]
            resid_wins = _make_windows(resid_np, window)
            score_wins = f_wins[:, :, None] * resid_wins  # (n_windows, window, N)

            xx = (f_wins**2).sum(axis=1)  # (n_windows,) — X'X per window

            # Gamma(0) plus Bartlett-weighted lags, summed over the window axis.
            S = np.einsum("twn,twn->tn", score_wins, score_wins) / window
            for lag in range(1, n_lags + 1):
                w = 1.0 - lag / (n_lags + 1)
                gamma = (
                    np.einsum(
                        "twn,twn->tn",
                        score_wins[:, lag:, :],
                        score_wins[:, : window - lag, :],
                    )
                    / window
                )
                S += 2 * w * gamma

            # Sandwich: Var(beta) = (X'X)^{-1} S (X'X)^{-1}, with S scaled by n_obs.
            var_beta = S * window / (xx[:, None] ** 2)
            se_vals = np.sqrt(np.maximum(var_beta, 0.0))  # (n_windows, N)

            # NaN masking (consistent with issue #8 / _fill_window):
            #   factor NaN  -> invalidate the whole window for every asset;
            #   residual NaN -> invalidate only the affected asset column.
            f_has_nan = np.isnan(f_wins).any(axis=1)  # (n_windows,)
            asset_has_nan = f_has_nan[:, None] | np.isnan(resid_wins).any(axis=1)
            t_idx = np.arange(n_windows) + window - 1
            se[t_idx] = np.where(asset_has_nan, np.nan, se_vals)

        if min_periods < window:
            # Early windows have variable size (< window) — keep the loop.
            for t in range(min_periods - 1, window - 1):
                _fill_window(t, f_np[: t + 1], resid_np[: t + 1])

    return pd.DataFrame(se, index=residuals.index, columns=residuals.columns)
