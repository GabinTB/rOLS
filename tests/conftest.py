"""Shared data generators and differential comparison helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def panel_factory() -> Callable[..., tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]]:
    """Generate deterministic panels with controllable correlation and missingness."""

    def make_panel(
        n_observations: int = 80,
        n_targets: int = 2,
        n_factors: int = 2,
        n_controls: int = 1,
        correlation: float = 0.5,
        nonzero_means: bool = True,
        nan_pattern: str = "none",
        near_collinear: bool = False,
        seed: int = 0,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
        if not -1 <= correlation <= 1:
            raise ValueError("correlation must lie in [-1, 1]")
        valid_patterns = {"none", "target-only", "regressor-only", "both", "structural-gap"}
        if nan_pattern not in valid_patterns:
            raise ValueError(f"nan_pattern must be one of {sorted(valid_patterns)}")

        rng = np.random.default_rng(seed)
        index = pd.date_range("2020-01-01", periods=n_observations, freq="D")
        common = rng.normal(size=(n_observations, 1))
        independent_scale = np.sqrt(max(0.0, 1 - correlation**2))
        controls_array = correlation * common + independent_scale * rng.normal(
            size=(n_observations, n_controls)
        )
        factors_array = correlation * common + independent_scale * rng.normal(
            size=(n_observations, n_factors)
        )
        if near_collinear and n_factors >= 2:
            factors_array[:, 1] = factors_array[:, 0] + 1e-10 * rng.normal(size=n_observations)
        if nonzero_means:
            factors_array += np.arange(1, n_factors + 1)
            if n_controls:
                controls_array += np.arange(1, n_controls + 1) * 0.5

        factor_coef = rng.normal(size=(n_factors, n_targets))
        control_coef = rng.normal(size=(n_controls, n_targets))
        targets_array = factors_array @ factor_coef + controls_array @ control_coef
        targets_array += rng.normal(scale=0.1, size=targets_array.shape)
        if nonzero_means:
            targets_array += np.arange(1, n_targets + 1)

        factors = pd.DataFrame(
            factors_array, index=index, columns=[f"factor_{i}" for i in range(n_factors)]
        )
        controls = (
            None
            if n_controls == 0
            else pd.DataFrame(
                controls_array,
                index=index,
                columns=[f"control_{i}" for i in range(n_controls)],
            )
        )
        targets = pd.DataFrame(
            targets_array, index=index, columns=[f"target_{i}" for i in range(n_targets)]
        )

        gap_start = max(1, n_observations // 4)
        gap_stop = min(n_observations, gap_start + max(1, n_observations // 10))
        if nan_pattern in {"target-only", "both"}:
            targets.iloc[gap_start:gap_stop, 0] = np.nan
        if nan_pattern in {"regressor-only", "both"}:
            factors.iloc[gap_start:gap_stop, 0] = np.nan
        if nan_pattern == "structural-gap":
            edge = max(1, n_observations // 5)
            targets.iloc[:edge, 0] = np.nan
            targets.iloc[-edge:, 0] = np.nan

        return targets, factors, controls

    return make_panel


def _first_mismatch(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    quantity: str,
    factor: str,
    rtol: float,
    atol: float,
) -> None:
    pd.testing.assert_index_equal(actual.index, expected.index)
    pd.testing.assert_index_equal(actual.columns, expected.columns)
    actual_values = actual.to_numpy(dtype=np.float64)
    expected_values = expected.to_numpy(dtype=np.float64)
    both_nan = np.isnan(actual_values) & np.isnan(expected_values)
    close = np.isclose(actual_values, expected_values, rtol=rtol, atol=atol, equal_nan=False)
    mismatch = ~(both_nan | close)
    if mismatch.any():
        row, column = np.argwhere(mismatch)[0]
        actual_value = actual_values[row, column].item()
        expected_value = expected_values[row, column].item()
        raise AssertionError(
            f"oracle mismatch for {quantity}, factor={factor}, "
            f"t={actual.index[row]!r}, target={actual.columns[column]!r}: "
            f"actual={actual_value!r}, expected={expected_value!r}"
        )


def _result_frame(result: object, quantity: str, factor: str) -> pd.DataFrame:
    if isinstance(result, Mapping):
        return result[quantity][factor]
    getter_name = {
        "beta": "get_beta",
        "intercept": "get_intercept",
        "residuals": "get_residuals",
        "r2": "get_r2",
        "adj_r2": "get_adj_r2",
        "dof": "get_dof",
        "n_used": "get_n_used",
        "n_eff": "get_n_eff",
    }[quantity]
    getter = getattr(result, getter_name, None)
    if getter is None:
        raise AssertionError(f"result does not report required quantity {quantity!r}")
    return getter(factor)


def assert_matches_oracle(
    result: object,
    oracle_out: Mapping[str, object],
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> None:
    """Compare every oracle quantity, failing on the first value or NaN mismatch."""
    quantities = ("beta", "intercept", "residuals", "r2", "adj_r2", "dof", "n_used", "n_eff")
    for quantity in quantities:
        expected_by_factor = oracle_out[quantity]
        for factor, expected in expected_by_factor.items():
            actual = _result_frame(result, quantity, factor)
            _first_mismatch(actual, expected, quantity, factor, rtol, atol)

    for factor, expected_by_control in oracle_out.get("control_beta", {}).items():
        for control, expected in expected_by_control.items():
            if isinstance(result, Mapping):
                actual = result["control_beta"][factor][control]
            else:
                getter = getattr(result, "get_control_beta", None)
                if getter is None:
                    raise AssertionError("result does not report required quantity 'control_beta'")
                actual = getter(factor, control)
            _first_mismatch(actual, expected, f"control_beta[{control}]", factor, rtol, atol)
