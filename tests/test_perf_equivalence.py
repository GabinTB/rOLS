"""Equivalence checks for missingness grouping and the within-window FWL path."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rols import RollingOLS
from rols.estimators import _group_mask_columns, rolling_fwl_solve, rolling_joint_solve


def _panel(
    n_controls: int,
    nan_pattern: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    rng = np.random.default_rng(1200 + n_controls)
    index = pd.RangeIndex(24)
    factors = pd.DataFrame(rng.normal(size=(24, 2)), index=index, columns=["f0", "f1"])
    controls = None
    if n_controls:
        controls = pd.DataFrame(
            rng.normal(size=(24, n_controls)),
            index=index,
            columns=[f"c{i}" for i in range(n_controls)],
        )
    targets = pd.DataFrame(
        rng.normal(size=(24, 5)),
        index=index,
        columns=[f"a{i}" for i in range(5)],
    )
    if nan_pattern == "structural":
        targets.loc[:3, "a0"] = np.nan
        targets.loc[20:, "a1"] = np.nan
        targets.loc[:3, "a2"] = np.nan
    elif nan_pattern == "scattered":
        for column, row in enumerate((2, 5, 8, 11, 14)):
            targets.iloc[row, column] = np.nan
    return targets, factors, controls


def _assert_fwl_matches_joint(
    targets: pd.DataFrame,
    factors: pd.DataFrame,
    controls: pd.DataFrame | None,
    *,
    fit_intercept: bool,
    expanding: bool,
) -> None:
    fwl = rolling_fwl_solve(
        targets,
        factors,
        controls,
        window=10,
        min_periods=7,
        expanding=expanding,
        fit_intercept=fit_intercept,
        warn_singular=False,
    )
    for factor_position, factor_name in enumerate(factors.columns):
        design_parts = [] if controls is None else [controls]
        design = pd.concat([*design_parts, factors[[factor_name]]], axis=1)
        joint = rolling_joint_solve(
            targets,
            design,
            window=10,
            min_periods=7,
            expanding=expanding,
            fit_intercept=fit_intercept,
            warn_singular=False,
        )
        np.testing.assert_allclose(
            fwl.factor_coef[:, factor_position], joint.coef[:, -1], rtol=0, atol=1e-10
        )
        np.testing.assert_allclose(
            fwl.intercept[:, factor_position], joint.intercept, rtol=0, atol=1e-10
        )
        np.testing.assert_allclose(
            fwl.resid_endpoint[:, factor_position],
            joint.resid_endpoint,
            rtol=0,
            atol=1e-10,
        )
        np.testing.assert_allclose(fwl.ssr[:, factor_position], joint.ssr, rtol=0, atol=1e-10)
        np.testing.assert_allclose(fwl.sst[:, factor_position], joint.sst, rtol=0, atol=1e-10)
        np.testing.assert_array_equal(fwl.n_used[:, factor_position], joint.n_used)
        np.testing.assert_allclose(fwl.n_eff[:, factor_position], joint.n_eff, rtol=0, atol=1e-12)
        if controls is not None:
            np.testing.assert_allclose(
                fwl.nuisance_coef[:, factor_position],
                joint.coef[:, : controls.shape[1]],
                rtol=0,
                atol=1e-10,
            )

    if controls is not None:
        nuisance_only = rolling_joint_solve(
            targets,
            controls,
            window=10,
            min_periods=7,
            expanding=expanding,
            fit_intercept=fit_intercept,
            warn_singular=False,
        )
        np.testing.assert_allclose(
            fwl.nuisance_resid_endpoint,
            nuisance_only.resid_endpoint,
            rtol=0,
            atol=1e-10,
        )


@pytest.mark.parametrize("n_controls", [0, 1, 3])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("nan_pattern", ["clean", "structural", "scattered"])
@pytest.mark.parametrize("expanding", [False, True])
def test_fwl_matches_joint_across_supported_configurations(
    n_controls: int,
    fit_intercept: bool,
    nan_pattern: str,
    expanding: bool,
) -> None:
    targets, factors, controls = _panel(n_controls, nan_pattern)
    _assert_fwl_matches_joint(
        targets,
        factors,
        controls,
        fit_intercept=fit_intercept,
        expanding=expanding,
    )


def test_lambda_routes_between_fwl_and_joint() -> None:
    targets, factors, controls = _panel(1, "clean")

    assert (
        RollingOLS(window=10, lambda_=0, mode="batched")
        .fit_transform(factors, targets, controls)
        ._path
        == "fwl"
    )
    assert (
        RollingOLS(window=10, lambda_=0.1, mode="batched")
        .fit_transform(factors, targets, controls)
        ._path
        == "joint"
    )


def test_pattern_grouping_is_exact() -> None:
    patterns = np.array(
        [
            [True, True, False, True, False],
            [True, True, True, False, True],
            [False, False, True, True, True],
            [True, True, False, True, False],
        ]
    )

    groups = _group_mask_columns(patterns)

    assert len(groups) == 3
    for group in groups:
        reference = patterns[:, group[0]]
        for position in group:
            np.testing.assert_array_equal(patterns[:, position], reference)


def test_shared_pattern_is_bitwise_column_permutation_invariant() -> None:
    targets, factors, controls = _panel(1, "clean")
    order = ["a3", "a0", "a4", "a1", "a2"]
    original = RollingOLS(window=10, mode="batched").fit_transform(
        factors, targets, controls, return_control_betas=True
    )
    permuted = RollingOLS(window=10, mode="batched").fit_transform(
        factors, targets[order], controls, return_control_betas=True
    )

    for factor in factors:
        for accessor in (
            "get_beta",
            "get_intercept",
            "get_residuals",
            "get_r2",
            "get_partial_r2",
            "get_n_used",
        ):
            expected = getattr(original, accessor)(factor)
            actual = getattr(permuted, accessor)(factor)[targets.columns]
            np.testing.assert_array_equal(actual, expected)
        assert controls is not None
        for control in controls:
            expected = original.get_control_beta(factor, control)
            actual = permuted.get_control_beta(factor, control)[targets.columns]
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("n_factors,n_targets", [(1, 1), (1, 4), (3, 1)])
def test_singleton_factor_and_target_cases(n_factors: int, n_targets: int) -> None:
    rng = np.random.default_rng(44)
    factors = pd.DataFrame(rng.normal(size=(20, n_factors)))
    targets = pd.DataFrame(rng.normal(size=(20, n_targets)))

    _assert_fwl_matches_joint(
        targets,
        factors,
        None,
        fit_intercept=True,
        expanding=False,
    )


def test_every_target_unique_pattern_matches_joint() -> None:
    targets, factors, controls = _panel(3, "scattered")

    _assert_fwl_matches_joint(
        targets,
        factors,
        controls,
        fit_intercept=True,
        expanding=False,
    )
