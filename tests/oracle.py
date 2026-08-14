"""Deliberately slow scalar reference implementation for rOLS tests.

The functions in this module favor direct correspondence with the statistical
specification over performance. They must never be imported by production code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OracleFit:
    """Results from one complete-case window fit for one target."""

    coef: np.ndarray
    intercept: float
    fitted: np.ndarray
    resid: np.ndarray
    ssr: float
    sst: float
    n_used: int
    n_eff: float


def _nan_fit(n_observations: int, n_slopes: int) -> OracleFit:
    return OracleFit(
        coef=np.full(n_slopes, np.nan),
        intercept=np.nan,
        fitted=np.full(n_observations, np.nan),
        resid=np.full(n_observations, np.nan),
        ssr=np.nan,
        sst=np.nan,
        n_used=0,
        n_eff=np.nan,
    )


def _normalized_complete_case_weights(
    weights: np.ndarray | None,
    complete_case: np.ndarray,
) -> np.ndarray:
    """Return weights on the complete-case sample, normalized to sum to one."""
    n_observations = complete_case.size
    if weights is None:
        complete_weights = np.ones(int(complete_case.sum()), dtype=np.float64)
    else:
        weights_array = np.asarray(weights, dtype=np.float64)
        if weights_array.shape != (n_observations,):
            raise ValueError(f"weights must have shape ({n_observations},)")
        if not np.isfinite(weights_array).all() or (weights_array < 0).any():
            raise ValueError("weights must be finite and non-negative")
        complete_weights = weights_array[complete_case]

    weight_sum = complete_weights.sum()
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("weights on the complete-case sample must have positive sum")
    return complete_weights / weight_sum


def oracle_fit_window(
    y: np.ndarray,
    X: np.ndarray,
    fit_intercept: bool,
    weights: np.ndarray | None,
    penalty: np.ndarray | None,
) -> OracleFit:
    """Fit complete-case OLS, WLS, or Ridge for one window and one target.

    ``X`` excludes the intercept. Observation weights are normalized after
    complete-case filtering. A nonzero diagonal penalty standardizes that
    regressor under the same weights before solving, then coefficients are
    returned in the original regressor units. The intercept is never penalized.
    Singular systems return a NaN-filled result.
    """
    target = np.asarray(y, dtype=np.float64)
    regressors = np.asarray(X, dtype=np.float64)
    if target.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if regressors.ndim != 2 or regressors.shape[0] != target.size:
        raise ValueError("X must be two-dimensional with the same row count as y")

    n_observations, n_slopes = regressors.shape
    n_parameters = n_slopes + int(fit_intercept)
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
    if fit_intercept and penalty_matrix[0, 0] != 0:
        raise ValueError("the intercept cannot be penalized")

    complete_case = np.isfinite(target) & np.isfinite(regressors).all(axis=1)
    if not complete_case.any():
        return _nan_fit(n_observations, n_slopes)

    complete_target = target[complete_case]
    complete_regressors = regressors[complete_case]
    complete_weights = _normalized_complete_case_weights(weights, complete_case)

    if fit_intercept:
        solve_design = np.column_stack(
            [np.ones(complete_target.size, dtype=np.float64), complete_regressors]
        )
    else:
        solve_design = complete_regressors.copy()

    means = np.zeros(n_parameters, dtype=np.float64)
    scales = np.ones(n_parameters, dtype=np.float64)
    penalty_diagonal = np.diag(penalty_matrix)
    penalized_columns = np.flatnonzero(penalty_diagonal > 0)
    for column in penalized_columns:
        values = solve_design[:, column]
        if fit_intercept:
            means[column] = np.sum(complete_weights * values)
            centered = values - means[column]
            scale_squared = np.sum(complete_weights * centered**2)
            solve_design[:, column] = centered
        else:
            scale_squared = np.sum(complete_weights * values**2)
        scales[column] = np.sqrt(scale_squared)
        if not np.isfinite(scales[column]) or scales[column] <= np.finfo(float).eps:
            return _nan_fit(n_observations, n_slopes)
        solve_design[:, column] /= scales[column]

    sqrt_weights = np.sqrt(complete_weights)
    weighted_design = solve_design * sqrt_weights[:, None]
    weighted_target = complete_target * sqrt_weights
    if penalized_columns.size:
        augmented_design = np.vstack([weighted_design, np.sqrt(penalty_matrix)])
        augmented_target = np.concatenate([weighted_target, np.zeros(n_parameters)])
    else:
        augmented_design = weighted_design
        augmented_target = weighted_target

    if np.linalg.matrix_rank(augmented_design) < n_parameters:
        return _nan_fit(n_observations, n_slopes)
    solve_coef = np.linalg.lstsq(augmented_design, augmented_target, rcond=None)[0]

    original_coef = solve_coef / scales
    if fit_intercept:
        intercept = original_coef[0] - np.sum(original_coef[1:] * means[1:])
        slopes = original_coef[1:]
    else:
        intercept = 0.0
        slopes = original_coef

    fitted_complete = intercept + complete_regressors @ slopes
    residual_complete = complete_target - fitted_complete
    fitted = np.full(n_observations, np.nan)
    residuals = np.full(n_observations, np.nan)
    fitted[complete_case] = fitted_complete
    residuals[complete_case] = residual_complete

    ssr = float(np.sum(complete_weights * residual_complete**2))
    if fit_intercept:
        target_mean = np.sum(complete_weights * complete_target)
        sst = float(np.sum(complete_weights * (complete_target - target_mean) ** 2))
    else:
        sst = float(np.sum(complete_weights * complete_target**2))
    n_eff = float(1.0 / np.sum(complete_weights**2))
    return OracleFit(
        coef=slopes,
        intercept=float(intercept),
        fitted=fitted,
        resid=residuals,
        ssr=ssr,
        sst=sst,
        n_used=int(complete_case.sum()),
        n_eff=n_eff,
    )


def _empty_factor_frames(
    factor_names: list[str],
    index: pd.Index,
    target_names: pd.Index,
) -> dict[str, pd.DataFrame]:
    return {
        factor: pd.DataFrame(np.nan, index=index, columns=target_names, dtype=np.float64)
        for factor in factor_names
    }


def oracle_rolling(
    targets: pd.DataFrame,
    factors: pd.DataFrame,
    controls: pd.DataFrame | None,
    window: int,
    min_periods: int,
    expanding: bool,
    fit_intercept: bool = True,
    lambda_: float = 0.0,
    ewma_halflife: int | None = None,
    mode: str = "batched",
    penalize_controls: bool = True,
) -> dict[str, object]:
    """Walk every endpoint and target using one scalar fit per selected model.

    In ``batched`` mode each factor is fitted separately with the controls. In
    ``joint`` mode all factors share one fit. Every output for a model is
    derived from that same complete-case window fit.
    """
    if mode not in {"batched", "joint"}:
        raise ValueError("mode must be 'batched' or 'joint'")
    if window <= 0 or min_periods <= 0:
        raise ValueError("window and min_periods must be positive")
    if not expanding and min_periods > window:
        raise ValueError("min_periods cannot exceed window for rolling estimation")
    if lambda_ < 0:
        raise ValueError("lambda_ must be non-negative")
    if ewma_halflife is not None and ewma_halflife <= 0:
        raise ValueError("ewma_halflife must be positive")
    if not targets.index.equals(factors.index):
        raise ValueError("targets and factors must have identical indexes")
    if controls is not None and not targets.index.equals(controls.index):
        raise ValueError("targets and controls must have identical indexes")

    factor_names = list(factors.columns)
    control_names = [] if controls is None else list(controls.columns)
    quantities = ["beta", "intercept", "residuals", "r2", "adj_r2", "dof", "n_used", "n_eff"]
    output: dict[str, object] = {
        quantity: _empty_factor_frames(factor_names, targets.index, targets.columns)
        for quantity in quantities
    }
    output["control_beta"] = {
        factor: {
            control: pd.DataFrame(
                np.nan, index=targets.index, columns=targets.columns, dtype=np.float64
            )
            for control in control_names
        }
        for factor in factor_names
    }

    model_factors = [[factor] for factor in factor_names]
    if mode == "joint":
        model_factors = [factor_names]

    target_values = targets.to_numpy(dtype=np.float64)
    control_values = (
        np.empty((len(targets), 0), dtype=np.float64)
        if controls is None
        else controls.to_numpy(dtype=np.float64)
    )
    factor_values = factors.to_numpy(dtype=np.float64)
    factor_positions = {factor: position for position, factor in enumerate(factor_names)}

    for selected_factors in model_factors:
        selected_positions = [factor_positions[factor] for factor in selected_factors]
        all_regressors = np.column_stack([control_values, factor_values[:, selected_positions]])
        n_slopes = all_regressors.shape[1]
        penalty = np.zeros((n_slopes + int(fit_intercept),) * 2, dtype=np.float64)
        penalty_offset = int(fit_intercept)
        if lambda_ > 0:
            if penalize_controls:
                penalty[
                    penalty_offset : penalty_offset + len(control_names),
                    penalty_offset : penalty_offset + len(control_names),
                ] = np.eye(len(control_names)) * lambda_
            factor_offset = penalty_offset + len(control_names)
            penalty[factor_offset:, factor_offset:] = np.eye(len(selected_factors)) * lambda_

        for endpoint in range(len(targets)):
            start = 0 if expanding else max(0, endpoint - window + 1)
            window_slice = slice(start, endpoint + 1)
            window_length = endpoint - start + 1
            raw_weights = None
            if ewma_halflife is not None:
                alpha = 1 - 2 ** (-1 / ewma_halflife)
                raw_weights = (1 - alpha) ** np.arange(window_length - 1, -1, -1)

            for target_position, target_name in enumerate(targets.columns):
                window_target = target_values[window_slice, target_position]
                window_regressors = all_regressors[window_slice]
                complete_count = int(
                    (np.isfinite(window_target) & np.isfinite(window_regressors).all(axis=1)).sum()
                )
                if complete_count < min_periods:
                    continue

                fit = oracle_fit_window(
                    window_target,
                    window_regressors,
                    fit_intercept=fit_intercept,
                    weights=raw_weights,
                    penalty=penalty,
                )
                if np.isnan(fit.coef).all():
                    continue

                r2 = np.nan if fit.sst <= 0 else 1 - fit.ssr / fit.sst
                residual_dof = fit.n_eff - n_slopes - int(fit_intercept)
                if residual_dof > 0 and np.isfinite(r2):
                    if fit_intercept:
                        adj_r2 = 1 - (1 - r2) * (fit.n_eff - 1) / residual_dof
                    else:
                        adj_r2 = 1 - (1 - r2) * fit.n_eff / residual_dof
                else:
                    adj_r2 = np.nan

                for factor_index, factor in enumerate(selected_factors):
                    factor_coef_index = len(control_names) + factor_index
                    output["beta"][factor].loc[targets.index[endpoint], target_name] = fit.coef[
                        factor_coef_index
                    ]
                    output["intercept"][factor].loc[targets.index[endpoint], target_name] = (
                        fit.intercept
                    )
                    output["residuals"][factor].loc[targets.index[endpoint], target_name] = (
                        fit.resid[-1]
                    )
                    output["r2"][factor].loc[targets.index[endpoint], target_name] = r2
                    output["adj_r2"][factor].loc[targets.index[endpoint], target_name] = adj_r2
                    output["dof"][factor].loc[targets.index[endpoint], target_name] = residual_dof
                    output["n_used"][factor].loc[targets.index[endpoint], target_name] = fit.n_used
                    output["n_eff"][factor].loc[targets.index[endpoint], target_name] = fit.n_eff
                    for control_index, control in enumerate(control_names):
                        output["control_beta"][factor][control].loc[
                            targets.index[endpoint], target_name
                        ] = fit.coef[control_index]

    return output


def oracle_hac_se(
    y: np.ndarray,
    X: np.ndarray,
    coef: np.ndarray,
    fit_intercept: bool,
    weights: np.ndarray | None,
    n_lags: int,
) -> np.ndarray:
    """Compute the weighted Newey-West sandwich for one window fit.

    ``coef`` is expressed in the supplied design coordinates and includes the
    intercept as its first element when ``fit_intercept=True``.
    """
    target = np.asarray(y, dtype=np.float64)
    regressors = np.asarray(X, dtype=np.float64)
    coefficients = np.asarray(coef, dtype=np.float64)
    if target.ndim != 1 or regressors.ndim != 2 or regressors.shape[0] != target.size:
        raise ValueError("y and X must have shapes (n,) and (n, p)")
    n_parameters = regressors.shape[1] + int(fit_intercept)
    if coefficients.shape != (n_parameters,):
        raise ValueError(f"coef must have shape ({n_parameters},)")
    if n_lags < 0:
        raise ValueError("n_lags must be non-negative")

    complete_case = np.isfinite(target) & np.isfinite(regressors).all(axis=1)
    complete_target = target[complete_case]
    complete_regressors = regressors[complete_case]
    if complete_target.size == 0:
        return np.full(n_parameters, np.nan)
    if n_lags >= complete_target.size:
        raise ValueError("n_lags must be smaller than the complete-case sample size")
    complete_weights = _normalized_complete_case_weights(weights, complete_case)
    design = (
        np.column_stack([np.ones(complete_target.size), complete_regressors])
        if fit_intercept
        else complete_regressors
    )
    n_eff = 1.0 / np.sum(complete_weights**2)
    if n_eff <= n_parameters:
        return np.full(n_parameters, np.nan)

    bread = design.T @ (complete_weights[:, None] * design)
    if np.linalg.matrix_rank(bread) < n_parameters:
        return np.full(n_parameters, np.nan)
    residuals = complete_target - design @ coefficients
    scores = complete_weights[:, None] * design * residuals[:, None]
    meat = scores.T @ scores
    for lag in range(1, n_lags + 1):
        bartlett_weight = 1 - lag / (n_lags + 1)
        lag_cross_product = scores[lag:].T @ scores[:-lag]
        meat += bartlett_weight * (lag_cross_product + lag_cross_product.T)

    inverse_bread = np.linalg.inv(bread)
    covariance = n_eff / (n_eff - n_parameters) * inverse_bread @ meat @ inverse_bread
    variances = np.diag(covariance)
    return np.sqrt(np.where(variances >= 0, variances, np.nan))
