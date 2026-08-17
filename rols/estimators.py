"""
rOLS estimator functions
=============

Low-level rolling estimators. All functions operate on numpy arrays
or pandas DataFrames and are independent of the model class.

Functions
---------
rolling_residualize   : rolling OLS/Ridge residualization (Frisch-Waugh step)
rolling_hac_se        : current-window Newey-West HAC standard errors
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


@dataclass(frozen=True)
class JointWindowResult:
    """One complete-case joint fit and optional current-window HAC output."""

    parameters: np.ndarray
    endpoint_residual: np.ndarray
    ssr: np.ndarray
    sst: np.ndarray
    n_used: int
    n_eff: float
    factor_se: np.ndarray | None = None
    hac_residuals: np.ndarray | None = None
    hac_bread_invalid: bool = False
    n_invalid_hac_variance: int = 0


@dataclass(frozen=True)
class BatchedFitResult:
    """Outputs from independent per-factor fits sharing one nuisance design."""

    factor_coef: np.ndarray | None
    intercept: np.ndarray | None
    n_used: np.ndarray | None
    sufficient_statistics: tuple[PatternSufficientStatistics, ...]
    nuisance_coef: np.ndarray | None = None
    nuisance_resid_endpoint: np.ndarray | None = None
    resid_endpoint: np.ndarray | None = None
    ssr: np.ndarray | None = None
    sst: np.ndarray | None = None
    reduced_ssr: np.ndarray | None = None
    n_eff: np.ndarray | None = None
    n_singular: int = 0
    n_ill_conditioned: int = 0


@dataclass(frozen=True)
class PatternSufficientStatistics:
    """FWL statistics for one endpoint and exact complete-case pattern."""

    endpoint: int
    factor_positions: np.ndarray
    target_positions: np.ndarray
    denominators: np.ndarray
    reduced_ssr: np.ndarray
    raw_sst: np.ndarray
    n_used: int
    n_eff: float


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


def _warn_invalid_hac(n_bread: int, n_variance: int) -> None:
    """Emit one warning for undefined HAC bread or non-positive variances."""
    if n_bread <= 0 and n_variance <= 0:
        return
    warnings.warn(
        "HAC inference returned NaN for "
        f"{n_bread} singular or near-singular bread block(s) and "
        f"{n_variance} non-positive variance estimate(s).",
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


def _group_mask_columns(mask: np.ndarray) -> list[np.ndarray]:
    """Group bitwise-identical boolean columns using packed byte keys."""
    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    packed = np.packbits(mask, axis=0)
    groups_by_key: dict[bytes, list[int]] = {}
    for position in range(mask.shape[1]):
        groups_by_key.setdefault(packed[:, position].tobytes(), []).append(position)

    groups = [np.asarray(positions, dtype=np.intp) for positions in groups_by_key.values()]
    for positions in groups:
        reference = mask[:, positions[0]]
        assert np.equal(mask[:, positions], reference[:, None]).all(), (
            "packed mask collision grouped unequal target masks"
        )
    return groups


def _selected_endpoint_pairs(
    endpoint_positions: np.ndarray | None,
    n_observations: int,
    min_periods: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Map full-index endpoints to compact output positions."""
    if endpoint_positions is None:
        endpoints = np.arange(min_periods - 1, n_observations, dtype=np.intp)
        return endpoints, endpoints, n_observations
    endpoints = np.asarray(endpoint_positions)
    if endpoints.ndim != 1 or not np.issubdtype(endpoints.dtype, np.integer):
        raise ValueError("endpoint_positions must be a one-dimensional integer array")
    endpoints = endpoints.astype(np.intp, copy=False)
    if endpoints.size and (
        endpoints[0] < min_periods - 1
        or endpoints[-1] >= n_observations
        or np.any(np.diff(endpoints) <= 0)
    ):
        raise ValueError(
            "endpoint_positions must be strictly increasing and within valid endpoint bounds"
        )
    return endpoints, np.arange(endpoints.size, dtype=np.intp), endpoints.size


def _effective_sample_size(weights: np.ndarray) -> float:
    """Scale-invariant Kish effective sample size for positive weight mass."""
    weight_mass = float(np.sum(weights))
    squared_mass = float(np.sum(weights**2))
    if weight_mass <= 0 or squared_mass <= 0:
        return np.nan
    return weight_mass**2 / squared_mass


def _sqrt_hac_variances(variances: np.ndarray) -> tuple[np.ndarray, int]:
    """Take guarded square roots and count finite non-positive variances."""
    standard_errors = np.full(variances.shape, np.nan)
    invalid_variance = np.isfinite(variances) & (variances <= 0)
    positive_variance = np.isfinite(variances) & (variances > 0)
    standard_errors[positive_variance] = np.sqrt(np.maximum(variances[positive_variance], 0.0))
    return standard_errors, int(invalid_variance.sum())


def _factor_hac_standard_errors(
    solve_design: np.ndarray,
    residuals: np.ndarray,
    complete_weights: np.ndarray,
    bread: np.ndarray,
    scales: np.ndarray,
    n_eff: float,
    n_lags: int,
    denom_tol: float,
) -> tuple[np.ndarray, bool, int]:
    """Return current-window HAC SEs for the final slope in solve coordinates."""
    n_observations, n_parameters = solve_design.shape
    standard_errors = np.full(residuals.shape[1], np.nan)
    if n_lags >= n_observations or n_eff <= n_parameters:
        return standard_errors, False, 0

    try:
        inverse_bread = np.linalg.solve(bread, np.eye(n_parameters))
    except np.linalg.LinAlgError:
        return standard_errors, True, 0
    factor_inverse_row = inverse_bread[-1]
    inverse_factor_information = inverse_bread[-1, -1]
    if not np.isfinite(inverse_factor_information) or inverse_factor_information <= 0:
        return standard_errors, True, 0
    factor_information = 1.0 / inverse_factor_information
    if not np.isfinite(factor_information) or factor_information <= denom_tol:
        return standard_errors, True, 0

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        bread_projection = solve_design @ factor_inverse_row
    if not np.isfinite(bread_projection).all():
        return standard_errors, True, 0
    projected_scores = complete_weights[:, None] * bread_projection[:, None] * residuals
    variances = np.einsum("wn,wn->n", projected_scores, projected_scores)
    for lag in range(1, n_lags + 1):
        bartlett_weight = 1.0 - lag / (n_lags + 1)
        variances += (
            2.0
            * bartlett_weight
            * np.einsum(
                "wn,wn->n",
                projected_scores[lag:],
                projected_scores[:-lag],
            )
        )

    correction = n_eff / (n_eff - n_parameters)
    variances *= correction
    variances /= scales[-1] ** 2
    standard_errors, n_invalid_variance = _sqrt_hac_variances(variances)
    return standard_errors, False, n_invalid_variance


def _solve_joint_window_block(
    targets: np.ndarray,
    design: np.ndarray,
    complete_case: np.ndarray,
    fit_intercept: bool,
    weights: np.ndarray | None,
    penalty: np.ndarray,
    cond_warn_threshold: float,
    hac_lags: int | None = None,
    denom_tol: float = 0.0,
    return_hac_residuals: bool = False,
) -> tuple[JointWindowResult | None, bool]:
    """Solve one complete-case window for a block of identically masked targets."""
    n_used = int(complete_case.sum())
    if n_used == 0:
        return None, False

    complete_targets = targets[complete_case]
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
    weighted_targets = complete_targets * sqrt_weights[:, None]
    design_gram = weighted_design.T @ weighted_design
    condition_number = float(np.linalg.cond(design_gram))
    if np.any(penalty):
        augmented_design = np.vstack([weighted_design, np.diag(np.sqrt(np.diag(penalty)))])
        augmented_targets = np.vstack(
            [weighted_targets, np.zeros((solve_design.shape[1], targets.shape[1]))]
        )
    else:
        augmented_design = weighted_design
        augmented_targets = weighted_targets

    if np.linalg.matrix_rank(augmented_design) < solve_design.shape[1]:
        return None, False
    try:
        q, r = np.linalg.qr(augmented_design, mode="reduced")
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            solve_parameters = np.linalg.solve(r, q.T @ augmented_targets)
    except np.linalg.LinAlgError:
        return None, False
    if not np.isfinite(solve_parameters).all():
        return None, False

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        parameters = solve_parameters / scales[:, None]
        if fit_intercept:
            parameters[0] -= means[1:] @ parameters[1:]
    if not np.isfinite(parameters).all():
        return None, condition_number > cond_warn_threshold

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        residuals = complete_targets - complete_design @ parameters
    if not np.isfinite(residuals).all():
        return None, condition_number > cond_warn_threshold
    ssr = np.sum(complete_weights[:, None] * residuals**2, axis=0)
    if fit_intercept:
        target_means = np.sum(complete_weights[:, None] * complete_targets, axis=0)
        sst = np.sum(
            complete_weights[:, None] * (complete_targets - target_means) ** 2,
            axis=0,
        )
    else:
        sst = np.sum(complete_weights[:, None] * complete_targets**2, axis=0)
    endpoint_residual = np.full(targets.shape[1], np.nan)
    if complete_case[-1]:
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            endpoint_residual = targets[-1] - design[-1] @ parameters
    n_eff = _effective_sample_size(complete_weights)
    factor_se = None
    hac_bread_invalid = False
    n_invalid_hac_variance = 0
    if hac_lags is not None:
        factor_se, hac_bread_invalid, n_invalid_hac_variance = _factor_hac_standard_errors(
            solve_design=solve_design,
            residuals=residuals,
            complete_weights=complete_weights,
            bread=design_gram + penalty,
            scales=scales,
            n_eff=n_eff,
            n_lags=hac_lags,
            denom_tol=denom_tol,
        )
    result = JointWindowResult(
        parameters=parameters,
        endpoint_residual=endpoint_residual,
        ssr=ssr,
        sst=sst,
        n_used=n_used,
        n_eff=n_eff,
        factor_se=factor_se,
        hac_residuals=residuals.copy() if return_hac_residuals else None,
        hac_bread_invalid=hac_bread_invalid,
        n_invalid_hac_variance=n_invalid_hac_variance,
    )
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
    endpoint_positions: np.ndarray | None = None,
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
    selected_endpoints, output_positions, output_size = _selected_endpoint_pairs(
        endpoint_positions,
        n_observations,
        min_periods,
    )

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

    coef = np.full((output_size, n_slopes, n_targets), np.nan)
    intercept = np.full((output_size, n_targets), np.nan)
    resid_endpoint = np.full((output_size, n_targets), np.nan)
    ssr = np.full((output_size, n_targets), np.nan)
    sst = np.full((output_size, n_targets), np.nan)
    n_used = np.full((output_size, n_targets), np.nan)
    n_eff = np.full((output_size, n_targets), np.nan)

    def store_block(
        endpoint: int,
        target_positions: np.ndarray,
        solved: JointWindowResult,
    ) -> None:
        parameters = solved.parameters
        if fit_intercept:
            intercept[endpoint, target_positions] = parameters[0]
            coef[endpoint][:, target_positions] = parameters[1:]
        else:
            intercept[endpoint, target_positions] = 0.0
            coef[endpoint][:, target_positions] = parameters
        resid_endpoint[endpoint, target_positions] = solved.endpoint_residual
        ssr[endpoint, target_positions] = solved.ssr
        sst[endpoint, target_positions] = solved.sst
        n_used[endpoint, target_positions] = solved.n_used
        n_eff[endpoint, target_positions] = solved.n_eff

    def solve_grouped_endpoint(
        endpoint: int,
        output_position: int,
        start: int,
        target_positions: np.ndarray,
        endpoint_weights: np.ndarray | None,
    ) -> tuple[int, int]:
        """Solve one endpoint once per exact target-validity pattern."""
        if target_positions.size == 0:
            return 0, 0
        window_targets = target_values[start : endpoint + 1, target_positions]
        window_design = design[start : endpoint + 1]
        target_validity = np.isfinite(window_targets)
        design_validity = np.isfinite(window_design).all(axis=1)
        singular_count = 0
        ill_conditioned_count = 0
        for local_positions in _group_mask_columns(target_validity):
            grouped_targets = target_positions[local_positions]
            complete_case = target_validity[:, local_positions[0]] & design_validity
            if int(complete_case.sum()) < min_periods:
                continue
            solved, ill_conditioned = _solve_joint_window_block(
                window_targets[:, local_positions],
                window_design,
                complete_case,
                fit_intercept,
                endpoint_weights,
                penalty_matrix,
                cond_warn_threshold,
            )
            group_size = grouped_targets.size
            ill_conditioned_count += int(ill_conditioned) * group_size
            if solved is None:
                singular_count += group_size
            else:
                store_block(output_position, grouped_targets, solved)
        return singular_count, ill_conditioned_count

    vectorizable_rolling = not expanding and np.isfinite(design).all() and n_observations >= window
    n_singular = 0
    n_ill_conditioned = 0
    if vectorizable_rolling:
        clean_target_positions = np.flatnonzero(np.isfinite(target_values).all(axis=0))
        dirty_target_positions = np.flatnonzero(~np.isfinite(target_values).all(axis=0))
        full_window_mask = selected_endpoints >= window - 1
        full_endpoints = selected_endpoints[full_window_mask]
        full_output_positions = output_positions[full_window_mask]
        full_window_offsets = full_endpoints - window + 1
        all_design_windows = _make_windows(design, window)
        all_target_windows = _make_windows(target_values[:, clean_target_positions], window)
        if endpoint_positions is None:
            design_windows = all_design_windows
            target_windows = all_target_windows
        else:
            design_windows = all_design_windows[full_window_offsets]
            target_windows = all_target_windows[full_window_offsets]
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
                intercept[full_output_positions, target_position] = parameters[:, 0, local_position]
                coef[full_output_positions, :, target_position] = parameters[:, 1:, local_position]
            else:
                intercept[full_output_positions, target_position] = 0.0
                coef[full_output_positions, :, target_position] = parameters[:, :, local_position]
            resid_endpoint[full_output_positions, target_position] = residual_windows[
                :, -1, local_position
            ]
            ssr[full_output_positions, target_position] = window_ssr[:, local_position]
            sst[full_output_positions, target_position] = window_sst[:, local_position]
            n_used[full_output_positions, target_position] = window
            n_eff[full_output_positions, target_position] = _effective_sample_size(window_weights)

        early_mask = selected_endpoints < window - 1
        for endpoint, output_position in zip(
            selected_endpoints[early_mask],
            output_positions[early_mask],
            strict=True,
        ):
            endpoint_weights = (
                None if supplied_weights is None else supplied_weights[-(endpoint + 1) :]
            )
            singular, ill_conditioned = solve_grouped_endpoint(
                endpoint,
                output_position,
                0,
                np.arange(n_targets),
                endpoint_weights,
            )
            n_singular += singular
            n_ill_conditioned += ill_conditioned
        for endpoint, output_position in zip(
            full_endpoints,
            full_output_positions,
            strict=True,
        ):
            start = endpoint - window + 1
            endpoint_weights = supplied_weights
            singular, ill_conditioned = solve_grouped_endpoint(
                endpoint,
                output_position,
                start,
                dirty_target_positions,
                endpoint_weights,
            )
            n_singular += singular
            n_ill_conditioned += ill_conditioned
    else:
        for endpoint, output_position in zip(
            selected_endpoints,
            output_positions,
            strict=True,
        ):
            start = 0 if expanding else max(0, endpoint - window + 1)
            endpoint_weights = (
                None if supplied_weights is None else supplied_weights[-(endpoint - start + 1) :]
            )
            singular, ill_conditioned = solve_grouped_endpoint(
                endpoint,
                output_position,
                start,
                np.arange(n_targets),
                endpoint_weights,
            )
            n_singular += singular
            n_ill_conditioned += ill_conditioned

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


def rolling_fwl_solve(
    y: pd.DataFrame,
    factors: pd.DataFrame,
    controls: pd.DataFrame | None,
    window: int,
    min_periods: int,
    expanding: bool,
    fit_intercept: bool = True,
    weights: np.ndarray | None = None,
    warn_singular: bool = True,
    cond_warn_threshold: float = 1e10,
    params_only: bool = False,
    return_nuisance_coef: bool = True,
    residuals_only: bool = False,
    endpoint_positions: np.ndarray | None = None,
) -> BatchedFitResult:
    """Fit independent factors by exact within-window FWL projection.

    Targets and factors are grouped by bitwise-identical validity masks. For
    each pair of pattern groups, one nuisance QR residualizes every factor and
    target on the same complete-case window. Factor-target cross-products are
    then computed by one matrix multiplication.
    """
    assert y.index.equals(factors.index), "y and factors must have identical indexes"
    if controls is not None:
        assert y.index.equals(controls.index), "y and controls must have identical indexes"
    if window <= 0 or min_periods <= 0:
        raise ValueError("window and min_periods must be positive")
    if not expanding and min_periods > window:
        raise ValueError("min_periods cannot exceed window")
    if weights is not None and expanding:
        raise ValueError("weights are not supported with expanding=True")
    if not np.isfinite(cond_warn_threshold) or cond_warn_threshold <= 0:
        raise ValueError("cond_warn_threshold must be finite and positive")

    canonical_target_order = np.asarray(
        sorted(
            range(y.shape[1]),
            key=lambda position: (
                type(y.columns[position]).__name__,
                repr(y.columns[position]),
            ),
        ),
        dtype=np.intp,
    )
    target_values = y.iloc[:, canonical_target_order].to_numpy(dtype=np.float64)
    factor_values = factors.to_numpy(dtype=np.float64)
    control_values = (
        np.empty((len(y), 0), dtype=np.float64)
        if controls is None
        else controls.to_numpy(dtype=np.float64)
    )
    nuisance_values = (
        np.column_stack([np.ones(len(y)), control_values]) if fit_intercept else control_values
    )
    n_observations, n_targets = target_values.shape
    n_factors = factor_values.shape[1]
    n_controls = control_values.shape[1]
    selected_endpoints, output_positions, output_size = _selected_endpoint_pairs(
        endpoint_positions,
        n_observations,
        min_periods,
    )

    supplied_weights = None
    if weights is not None:
        supplied_weights = np.asarray(weights, dtype=np.float64)
        if supplied_weights.shape != (window,):
            raise ValueError(f"weights must have shape ({window},)")
        if not np.isfinite(supplied_weights).all() or (supplied_weights < 0).any():
            raise ValueError("weights must be finite and non-negative")
        if supplied_weights.sum() <= 0:
            raise ValueError("weights must have positive sum")

    shape = (output_size, n_factors, n_targets)
    factor_coef = None if residuals_only else np.full(shape, np.nan)
    intercept = None if residuals_only else np.full(shape, np.nan)
    n_used = None if residuals_only or params_only else np.full(shape, np.nan)
    resid_endpoint = np.full(shape, np.nan) if residuals_only or not params_only else None
    ssr = None if params_only else np.full(shape, np.nan)
    sst = None if params_only else np.full(shape, np.nan)
    reduced_ssr = None if params_only else np.full(shape, np.nan)
    n_eff = None if params_only else np.full(shape, np.nan)
    nuisance_coef = (
        np.full((output_size, n_factors, n_controls, n_targets), np.nan)
        if return_nuisance_coef
        else None
    )
    nuisance_resid_endpoint = None if params_only else np.full((output_size, n_targets), np.nan)
    sufficient_statistics: list[PatternSufficientStatistics] = []
    n_singular = 0
    n_ill_conditioned = 0

    for endpoint, output_position in zip(
        selected_endpoints,
        output_positions,
        strict=True,
    ):
        start = 0 if expanding else max(0, endpoint - window + 1)
        window_targets = target_values[start : endpoint + 1]
        window_factors = factor_values[start : endpoint + 1]
        window_nuisance = nuisance_values[start : endpoint + 1]
        target_validity = np.isfinite(window_targets)
        factor_validity = np.isfinite(window_factors)
        factor_groups = _group_mask_columns(factor_validity)
        nuisance_validity = np.isfinite(window_nuisance).all(axis=1)
        endpoint_weights = (
            None if supplied_weights is None else supplied_weights[-(endpoint - start + 1) :]
        )

        for target_positions in _group_mask_columns(target_validity):
            target_mask = target_validity[:, target_positions[0]]
            target_output_positions = canonical_target_order[target_positions]
            for factor_positions in factor_groups:
                complete_case = (
                    target_mask & factor_validity[:, factor_positions[0]] & nuisance_validity
                )
                observations_used = int(complete_case.sum())
                if observations_used < min_periods:
                    continue

                complete_targets = window_targets[complete_case][:, target_positions]
                complete_factors = window_factors[complete_case][:, factor_positions]
                complete_nuisance = window_nuisance[complete_case]
                if endpoint_weights is None:
                    complete_weights = np.full(observations_used, 1.0 / observations_used)
                else:
                    complete_weights = endpoint_weights[complete_case]
                    weight_sum = complete_weights.sum()
                    if weight_sum <= 0:
                        continue
                    complete_weights = complete_weights / weight_sum
                sqrt_weights = np.sqrt(complete_weights)
                weighted_targets = complete_targets * sqrt_weights[:, None]
                weighted_factors = complete_factors * sqrt_weights[:, None]
                weighted_nuisance = complete_nuisance * sqrt_weights[:, None]

                if weighted_nuisance.shape[1]:
                    if np.linalg.matrix_rank(weighted_nuisance) < weighted_nuisance.shape[1]:
                        n_singular += factor_positions.size * target_positions.size
                        continue
                    try:
                        nuisance_q, nuisance_r = np.linalg.qr(weighted_nuisance, mode="reduced")
                        projected_targets = np.sum(
                            nuisance_q[:, :, None] * weighted_targets[:, None, :],
                            axis=0,
                        )
                        projected_factors = np.sum(
                            nuisance_q[:, :, None] * weighted_factors[:, None, :],
                            axis=0,
                        )
                        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                            target_residuals = weighted_targets - nuisance_q @ projected_targets
                            factor_residuals = weighted_factors - nuisance_q @ projected_factors
                        nuisance_target_coef = np.linalg.solve(nuisance_r, projected_targets)
                        nuisance_factor_coef = np.linalg.solve(nuisance_r, projected_factors)
                    except np.linalg.LinAlgError:
                        n_singular += factor_positions.size * target_positions.size
                        continue
                else:
                    target_residuals = weighted_targets
                    factor_residuals = weighted_factors
                    nuisance_target_coef = np.empty((0, target_positions.size))
                    nuisance_factor_coef = np.empty((0, factor_positions.size))

                nuisance_complete_case = target_mask & nuisance_validity
                if (
                    nuisance_resid_endpoint is not None
                    and complete_case[-1]
                    and np.array_equal(complete_case, nuisance_complete_case)
                ):
                    nuisance_endpoint_fit = np.zeros(target_positions.size)
                    if weighted_nuisance.shape[1]:
                        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                            nuisance_endpoint_fit = complete_nuisance[-1] @ nuisance_target_coef
                    nuisance_resid_endpoint[output_position, target_output_positions] = (
                        window_targets[-1, target_positions] - nuisance_endpoint_fit
                    )

                n_group_factors = factor_positions.size
                n_nuisance = weighted_nuisance.shape[1]
                design_grams = np.zeros(
                    (n_group_factors, n_nuisance + 1, n_nuisance + 1),
                    dtype=np.float64,
                )
                if n_nuisance:
                    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                        nuisance_gram = weighted_nuisance.T @ weighted_nuisance
                        nuisance_factor_cross = weighted_nuisance.T @ weighted_factors
                    design_grams[:, :n_nuisance, :n_nuisance] = nuisance_gram
                    design_grams[:, :n_nuisance, -1] = nuisance_factor_cross.T
                    design_grams[:, -1, :n_nuisance] = nuisance_factor_cross.T
                design_grams[:, -1, -1] = np.sum(weighted_factors**2, axis=0)
                condition_numbers = np.linalg.cond(design_grams)
                rank_condition_limit = (
                    1.0 / (max(observations_used, n_nuisance + 1) * np.finfo(np.float64).eps)
                ) ** 2
                valid_factors = np.isfinite(condition_numbers) & (
                    condition_numbers < rank_condition_limit
                )
                n_singular += int((~valid_factors).sum()) * target_positions.size
                n_ill_conditioned += (
                    int((valid_factors & (condition_numbers > cond_warn_threshold)).sum())
                    * target_positions.size
                )
                if not valid_factors.any():
                    continue

                # One GEMM computes every factor-target cross-product in this
                # exact pair of validity-pattern groups.
                if target_positions.size == 1:
                    # A two-column RHS keeps the BLAS reduction identical when
                    # a target moves between a singleton and a shared group.
                    duplicated_targets = np.repeat(target_residuals, 2, axis=1)
                    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                        cross_products = (factor_residuals.T @ duplicated_targets)[:, :1]
                else:
                    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                        cross_products = factor_residuals.T @ target_residuals
                denominators = np.sum(factor_residuals**2, axis=0)
                betas = np.divide(
                    cross_products,
                    denominators[:, None],
                    out=np.full_like(cross_products, np.nan),
                    where=valid_factors[:, None] & (denominators[:, None] > 0),
                )
                full_nuisance_coef = (
                    nuisance_target_coef[:, None, :]
                    - nuisance_factor_coef[:, :, None] * betas[None, :, :]
                )
                target_residual_series = np.ascontiguousarray(target_residuals.T)
                group_reduced_ssr = np.sum(target_residual_series**2, axis=1)
                group_ssr = group_reduced_ssr[None, :] - betas**2 * denominators[:, None]
                if fit_intercept:
                    target_means = np.sum(
                        complete_weights[:, None] * complete_targets,
                        axis=0,
                    )
                    group_sst = np.sum(
                        complete_weights[:, None] * (complete_targets - target_means) ** 2,
                        axis=0,
                    )
                else:
                    group_sst = np.sum(
                        complete_weights[:, None] * complete_targets**2,
                        axis=0,
                    )

                factor_index, target_index = np.ix_(factor_positions, target_output_positions)
                valid_matrix = np.broadcast_to(
                    valid_factors[:, None],
                    (factor_positions.size, target_positions.size),
                )
                if factor_coef is not None and intercept is not None:
                    factor_coef[output_position][factor_index, target_index] = betas
                    if fit_intercept:
                        intercept[output_position][factor_index, target_index] = full_nuisance_coef[
                            0
                        ]
                    else:
                        intercept_values = np.where(valid_matrix, 0.0, np.nan)
                        intercept[output_position][factor_index, target_index] = intercept_values
                control_offset = int(fit_intercept)
                if n_controls and nuisance_coef is not None:
                    control_values_block = full_nuisance_coef[
                        control_offset : control_offset + n_controls
                    ].transpose(1, 0, 2)
                    nuisance_coef[output_position][
                        factor_positions[:, None, None],
                        np.arange(n_controls)[None, :, None],
                        target_output_positions[None, None, :],
                    ] = control_values_block

                metric_valid = np.where(valid_matrix, 1.0, np.nan)
                if n_used is not None:
                    n_used[output_position][factor_index, target_index] = (
                        observations_used * metric_valid
                    )
                effective_count = _effective_sample_size(complete_weights)
                if not residuals_only:
                    sufficient_statistics.append(
                        PatternSufficientStatistics(
                            endpoint=output_position,
                            factor_positions=factor_positions.copy(),
                            target_positions=target_output_positions.copy(),
                            denominators=denominators.copy(),
                            reduced_ssr=group_reduced_ssr.copy(),
                            raw_sst=group_sst.copy(),
                            n_used=observations_used,
                            n_eff=effective_count,
                        )
                    )

                if ssr is not None:
                    assert sst is not None and reduced_ssr is not None and n_eff is not None
                    ssr[output_position][factor_index, target_index] = group_ssr * metric_valid
                    sst[output_position][factor_index, target_index] = (
                        group_sst[None, :] * metric_valid
                    )
                    reduced_ssr[output_position][factor_index, target_index] = (
                        group_reduced_ssr[None, :] * metric_valid
                    )
                    n_eff[output_position][factor_index, target_index] = (
                        effective_count * metric_valid
                    )

                if resid_endpoint is not None and complete_case[-1]:
                    endpoint_targets = window_targets[-1, target_positions]
                    endpoint_factors = window_factors[-1, factor_positions]
                    fitted_endpoint = endpoint_factors[:, None] * betas
                    if n_nuisance:
                        fitted_endpoint += np.einsum(
                            "n,nft->ft",
                            complete_nuisance[-1],
                            full_nuisance_coef,
                        )
                    endpoint_residuals = endpoint_targets[None, :] - fitted_endpoint
                    resid_endpoint[output_position][factor_index, target_index] = endpoint_residuals

    if warn_singular:
        _warn_singular(n_singular)
        _warn_ill_conditioned(n_ill_conditioned, cond_warn_threshold)
    return BatchedFitResult(
        factor_coef=factor_coef,
        intercept=intercept,
        n_used=n_used,
        sufficient_statistics=tuple(sufficient_statistics),
        nuisance_coef=nuisance_coef,
        nuisance_resid_endpoint=nuisance_resid_endpoint,
        resid_endpoint=resid_endpoint,
        ssr=ssr,
        sst=sst,
        reduced_ssr=reduced_ssr,
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
# HAC (Newey-West) standard errors
# ---------------------------------------------------------------------------


def rolling_hac_se(
    y: pd.DataFrame,
    X: pd.DataFrame,
    window: int,
    min_periods: int,
    expanding: bool,
    n_lags: int,
    fit_intercept: bool = True,
    penalty: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    denom_tol: float = 1e-12,
    warn_invalid: bool = True,
    endpoint_positions: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute factor HAC SEs from each endpoint's own current-window fit.

    X contains the full slope design for one factor model, with the factor
    of interest in its final column. Each endpoint and exact target-validity
    pattern is solved independently. Its in-window residual block is reduced
    immediately to one factor standard error per target and then released.

    Bartlett lags count ordered complete-case observations. Supplied weights are
    restricted to those rows and normalized before both the weighted fit and
    sandwich. The small-sample correction is n_eff / (n_eff - k), where k
    includes the intercept when present.
    """
    assert y.index.equals(X.index), "y and X must have identical indexes"
    if window <= 0 or min_periods <= 0:
        raise ValueError("window and min_periods must be positive")
    if not expanding and min_periods > window:
        raise ValueError("min_periods cannot exceed window")
    if n_lags < 0:
        raise ValueError("n_lags must be non-negative")
    if weights is not None and expanding:
        raise ValueError("weights are not supported with expanding=True")
    if not np.isfinite(denom_tol) or denom_tol < 0:
        raise ValueError("denom_tol must be finite and non-negative")

    target_values = y.to_numpy(dtype=np.float64)
    regressor_values = X.to_numpy(dtype=np.float64)
    n_observations, n_targets = target_values.shape
    selected_endpoints, output_positions, output_size = _selected_endpoint_pairs(
        endpoint_positions,
        n_observations,
        min_periods,
    )
    design = (
        np.column_stack([np.ones(n_observations), regressor_values])
        if fit_intercept
        else regressor_values
    )
    n_parameters = design.shape[1]
    if regressor_values.shape[1] == 0:
        raise ValueError("X must contain at least one factor column")

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
        if fit_intercept and (
            not np.allclose(penalty_matrix[0], 0.0) or not np.allclose(penalty_matrix[:, 0], 0.0)
        ):
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

    standard_errors = np.full((output_size, n_targets), np.nan)
    n_invalid_bread = 0
    n_invalid_variance = 0
    for endpoint, output_position in zip(
        selected_endpoints,
        output_positions,
        strict=True,
    ):
        start = 0 if expanding else max(0, endpoint - window + 1)
        window_targets = target_values[start : endpoint + 1]
        window_design = design[start : endpoint + 1]
        target_validity = np.isfinite(window_targets)
        design_validity = np.isfinite(window_design).all(axis=1)
        endpoint_weights = (
            None if supplied_weights is None else supplied_weights[-(endpoint - start + 1) :]
        )

        for target_positions in _group_mask_columns(target_validity):
            complete_case = target_validity[:, target_positions[0]] & design_validity
            if int(complete_case.sum()) < min_periods:
                continue
            solved, _ = _solve_joint_window_block(
                targets=window_targets[:, target_positions],
                design=window_design,
                complete_case=complete_case,
                fit_intercept=fit_intercept,
                weights=endpoint_weights,
                penalty=penalty_matrix,
                cond_warn_threshold=np.inf,
                hac_lags=n_lags,
                denom_tol=denom_tol,
            )
            if solved is None:
                n_invalid_bread += target_positions.size
                continue
            if solved.hac_bread_invalid:
                n_invalid_bread += target_positions.size
            n_invalid_variance += solved.n_invalid_hac_variance
            if solved.factor_se is not None:
                standard_errors[output_position, target_positions] = solved.factor_se

    if warn_invalid:
        _warn_invalid_hac(n_invalid_bread, n_invalid_variance)
    output_index = y.index if endpoint_positions is None else y.index[selected_endpoints]
    return pd.DataFrame(standard_errors, index=output_index, columns=y.columns)
