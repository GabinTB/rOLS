"""Systematic differential sweep of RollingOLS against the scalar oracle (F22).

Parametrises across controls, intercept, mode, lambda_, weighting, window
type, cadence, and NaN pattern. Every reported quantity — beta, intercept,
residuals, r2, dof, n_used, and control betas — is checked against
tests.oracle.oracle_rolling at every endpoint the model actually reports
(cadence subsetting is respected: rows dropped by ``estimate_every`` are
skipped, not compared against a NaN oracle value).

A hand-picked subset that touches every axis value at least once runs by
default. The full cross product runs under ``pytest.mark.slow``.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from rols import RollingOLS
from tests.oracle import oracle_rolling

CONTROLS_AXIS = (0, 1, 3)
INTERCEPT_AXIS = (True, False)
MODE_AXIS = ("batched", "joint")
LAMBDA_AXIS = (0.0, 1e-3, 1.0)
WEIGHTING_AXIS = (False, True)  # False = equal weights, True = ewma_halflife=window//4
WINDOW_TYPE_AXIS = ("rolling", "expanding")
CADENCE_AXIS = (1, 5, "W-FRI")
NAN_PATTERN_AXIS = ("none", "target-only", "regressor-only", "both", "structural-gap")

N_OBSERVATIONS = 48
WINDOW = 16


def _combo_valid(combo: tuple) -> bool:
    _, _, _, _, use_ewma, window_type, _, _ = combo
    # ewma_halflife cannot combine with expanding=True (variable window
    # length has no fixed weight vector to precompute).
    return not (use_ewma and window_type == "expanding")


FULL_PRODUCT = [
    combo
    for combo in itertools.product(
        CONTROLS_AXIS,
        INTERCEPT_AXIS,
        MODE_AXIS,
        LAMBDA_AXIS,
        WEIGHTING_AXIS,
        WINDOW_TYPE_AXIS,
        CADENCE_AXIS,
        NAN_PATTERN_AXIS,
    )
    if _combo_valid(combo)
]

# Touches every axis value at least once; hand-picked (not sampled) so a
# failure reproduces deterministically without depending on RNG state.
REPRESENTATIVE_SUBSET = [
    (0, True, "batched", 0.0, False, "rolling", 1, "none"),
    (1, False, "batched", 1e-3, True, "rolling", 5, "target-only"),
    (3, True, "joint", 1.0, False, "expanding", "W-FRI", "regressor-only"),
    (1, True, "joint", 0.0, True, "rolling", 1, "both"),
    (3, False, "batched", 1e-3, False, "rolling", 5, "structural-gap"),
    (0, True, "joint", 1.0, True, "rolling", "W-FRI", "none"),
    (1, True, "batched", 0.0, False, "expanding", 1, "structural-gap"),
    (3, True, "joint", 1e-3, True, "rolling", 1, "target-only"),
]
assert all(_combo_valid(combo) for combo in REPRESENTATIVE_SUBSET)


def _assert_kept_rows_match(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    kept: np.ndarray,
    label: str,
) -> None:
    """Compare only rows the model actually reports (respects cadence)."""
    actual_values = actual.to_numpy(dtype=np.float64)
    expected_values = expected.to_numpy(dtype=np.float64)
    both_nan = np.isnan(actual_values) & np.isnan(expected_values)
    close = np.isclose(actual_values, expected_values, rtol=1e-6, atol=1e-8, equal_nan=False)
    ok = ~kept | both_nan | close
    if not ok.all():
        row, col = np.argwhere(~ok)[0]
        raise AssertionError(
            f"{label} mismatch at t={actual.index[row]!r}, target={actual.columns[col]!r}: "
            f"actual={actual_values[row, col]!r}, expected={expected_values[row, col]!r}"
        )


def _run_and_compare(
    n_controls: int,
    fit_intercept: bool,
    mode: str,
    lambda_: float,
    use_ewma: bool,
    window_type: str,
    cadence,
    nan_pattern: str,
    panel_factory,
) -> None:
    expanding = window_type == "expanding"
    min_periods = WINDOW // 2 if expanding else WINDOW
    ewma_halflife = max(1, WINDOW // 4) if use_ewma else None

    seed = abs(
        hash(
            (
                n_controls,
                fit_intercept,
                mode,
                lambda_,
                use_ewma,
                window_type,
                str(cadence),
                nan_pattern,
            )
        )
    ) % (2**31)
    targets, factors, controls = panel_factory(
        n_observations=N_OBSERVATIONS,
        n_targets=2,
        n_factors=2,
        n_controls=n_controls,
        correlation=0.5,
        nan_pattern=nan_pattern,
        seed=seed,
    )

    ols = RollingOLS(
        window=WINDOW,
        min_periods=min_periods,
        expanding=expanding,
        fit_intercept=fit_intercept,
        lambda_=lambda_,
        mode=mode,
        ewma_halflife=ewma_halflife,
        estimate_every=cadence,
        precision="double",
        warn_singular=False,
        warn_correlated_factors=False,
    )
    result = ols.fit_transform(
        factors, targets, controls=controls, return_control_betas=(n_controls > 0)
    )

    expected = oracle_rolling(
        targets,
        factors,
        controls,
        window=WINDOW,
        min_periods=min_periods,
        expanding=expanding,
        fit_intercept=fit_intercept,
        lambda_=lambda_,
        ewma_halflife=ewma_halflife,
        mode=mode,
    )

    for factor in factors.columns:
        actual_beta = result.get_beta(factor)
        kept = actual_beta.notna().to_numpy()
        _assert_kept_rows_match(actual_beta, expected["beta"][factor], kept, "beta")
        _assert_kept_rows_match(
            result.get_intercept(factor), expected["intercept"][factor], kept, "intercept"
        )
        _assert_kept_rows_match(
            result.get_residuals(factor), expected["residuals"][factor], kept, "residuals"
        )
        _assert_kept_rows_match(result.get_r2(factor), expected["r2"][factor], kept, "r2")
        _assert_kept_rows_match(result.get_dof(factor), expected["dof"][factor], kept, "dof")
        _assert_kept_rows_match(
            result.get_n_used(factor), expected["n_used"][factor], kept, "n_used"
        )
        if n_controls:
            for control in controls.columns:
                _assert_kept_rows_match(
                    result.get_control_beta(factor, control),
                    expected["control_beta"][factor][control],
                    kept,
                    f"control_beta[{control}]",
                )


@pytest.mark.parametrize(
    "n_controls,fit_intercept,mode,lambda_,use_ewma,window_type,cadence,nan_pattern",
    REPRESENTATIVE_SUBSET,
)
def test_differential_sweep_representative(
    n_controls,
    fit_intercept,
    mode,
    lambda_,
    use_ewma,
    window_type,
    cadence,
    nan_pattern,
    panel_factory,
):
    _run_and_compare(
        n_controls,
        fit_intercept,
        mode,
        lambda_,
        use_ewma,
        window_type,
        cadence,
        nan_pattern,
        panel_factory,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "n_controls,fit_intercept,mode,lambda_,use_ewma,window_type,cadence,nan_pattern",
    FULL_PRODUCT,
)
def test_differential_sweep_full_product(
    n_controls,
    fit_intercept,
    mode,
    lambda_,
    use_ewma,
    window_type,
    cadence,
    nan_pattern,
    panel_factory,
):
    _run_and_compare(
        n_controls,
        fit_intercept,
        mode,
        lambda_,
        use_ewma,
        window_type,
        cadence,
        nan_pattern,
        panel_factory,
    )
