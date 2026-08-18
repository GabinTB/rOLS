"""Lazy result identity, cache, and memory-budget tests."""

from __future__ import annotations

import gc
import tracemalloc

import numpy as np
import pandas as pd
import pytest

from rols import RollingOLS
from rols.estimators import (
    _effective_sample_size,
    rolling_fwl_solve,
    rolling_hac_se,
    rolling_joint_solve,
)
from tests.oracle import oracle_fit_window, oracle_rolling


def _small_panel(
    *,
    n_observations: int = 48,
    n_targets: int = 4,
    n_factors: int = 3,
    n_controls: int = 2,
    seed: int = 13,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n_observations)
    factors = pd.DataFrame(
        rng.normal(size=(n_observations, n_factors)),
        index=index,
        columns=[f"f{i}" for i in range(n_factors)],
    )
    controls = pd.DataFrame(
        rng.normal(size=(n_observations, n_controls)),
        index=index,
        columns=[f"c{i}" for i in range(n_controls)],
    )
    targets = pd.DataFrame(
        rng.normal(size=(n_observations, n_targets)),
        index=index,
        columns=[f"a{i}" for i in range(n_targets)],
    )
    targets.iloc[:3, 0] = np.nan
    targets.iloc[-5:, 1] = np.nan
    return targets, factors, controls


def test_lazy_accessors_match_eager_fwl_outputs() -> None:
    targets, factors, controls = _small_panel()
    window = 16
    min_periods = 11
    halflife = 5
    weights = np.exp(-np.log(2) / halflife * np.arange(window - 1, -1, -1))
    weights /= weights.sum()
    eager = rolling_fwl_solve(
        targets,
        factors,
        controls,
        window,
        min_periods,
        False,
        weights=weights,
        warn_singular=False,
    )
    result = RollingOLS(
        window=window,
        min_periods=min_periods,
        ewma_halflife=halflife,
        adj_r2=True,
        hac_lags=2,
        dtype="float64",
        mode="batched",
    ).fit_transform(factors, targets, controls, return_control_betas=True)

    assert eager.factor_coef is not None
    assert eager.intercept is not None
    assert eager.resid_endpoint is not None
    assert eager.ssr is not None
    assert eager.sst is not None
    assert eager.reduced_ssr is not None
    assert eager.n_used is not None
    assert eager.n_eff is not None
    assert eager.nuisance_coef is not None
    assert eager.nuisance_resid_endpoint is not None
    for factor_position, factor in enumerate(factors):
        beta = pd.DataFrame(
            eager.factor_coef[:, factor_position], index=targets.index, columns=targets.columns
        )
        intercept = pd.DataFrame(
            eager.intercept[:, factor_position], index=targets.index, columns=targets.columns
        )
        residuals = pd.DataFrame(
            eager.resid_endpoint[:, factor_position],
            index=targets.index,
            columns=targets.columns,
        )
        n_used = pd.DataFrame(
            eager.n_used[:, factor_position], index=targets.index, columns=targets.columns
        )
        residual_dof = eager.n_eff[:, factor_position] - controls.shape[1] - 2
        r2 = 1.0 - eager.ssr[:, factor_position] / eager.sst[:, factor_position]
        partial_r2 = (
            eager.reduced_ssr[:, factor_position] - eager.ssr[:, factor_position]
        ) / eager.reduced_ssr[:, factor_position]
        adjustment = (eager.n_eff[:, factor_position] - 1) / residual_dof
        r2 = 1.0 - (1.0 - r2) * adjustment
        partial_r2 = 1.0 - (1.0 - partial_r2) * adjustment
        signal = beta.mul(factors[factor], axis=0)
        design = pd.concat([controls, factors[[factor]]], axis=1)
        se = rolling_hac_se(
            targets,
            design,
            window,
            min_periods,
            False,
            2,
            weights=weights,
        )

        expected = {
            "get_beta": beta,
            "get_intercept": intercept,
            "get_signal": signal,
            "get_r2": pd.DataFrame(r2, index=targets.index, columns=targets.columns),
            "get_partial_r2": pd.DataFrame(
                partial_r2, index=targets.index, columns=targets.columns
            ),
            "get_residuals": residuals,
            "get_dof": pd.DataFrame(residual_dof, index=targets.index, columns=targets.columns),
            "get_n_used": n_used,
            "get_se": se,
            "get_tstat": beta.div(se),
        }
        for accessor, expected_values in expected.items():
            np.testing.assert_allclose(
                getattr(result, accessor)(factor), expected_values, rtol=0, atol=1e-12
            )
        for control_position, control in enumerate(controls):
            expected_control = pd.DataFrame(
                eager.nuisance_coef[:, factor_position, control_position],
                index=targets.index,
                columns=targets.columns,
            )
            np.testing.assert_allclose(
                result.get_control_beta(factor, control),
                expected_control,
                rtol=0,
                atol=1e-12,
            )

    np.testing.assert_allclose(
        result.get_factor_adjusted_returns(),
        eager.nuisance_resid_endpoint,
        rtol=0,
        atol=1e-12,
    )


def test_lazy_accessors_match_eager_joint_ridge_outputs() -> None:
    targets, factors, controls = _small_panel(n_factors=2)
    factors.iloc[9:12, 1] = np.nan
    model = RollingOLS(
        window=16,
        min_periods=11,
        lambda_=0.05,
        penalize_controls=True,
        hac_lags=2,
        dtype="float64",
        mode="batched",
    )
    result = model.fit_transform(factors, targets, controls, return_control_betas=True)
    penalty = model._penalty_matrix(controls.shape[1], n_factors=1)
    controls_penalty = model._penalty_matrix(controls.shape[1], n_factors=0)

    for factor in factors:
        design = pd.concat([controls, factors[[factor]]], axis=1)
        eager = rolling_joint_solve(
            targets,
            design,
            window=16,
            min_periods=11,
            expanding=False,
            penalty=penalty,
            warn_singular=False,
        )
        factor_valid = np.isfinite(factors[factor])
        reduced = rolling_joint_solve(
            targets.where(factor_valid, axis=0),
            controls,
            window=16,
            min_periods=11,
            expanding=False,
            penalty=controls_penalty,
            warn_singular=False,
        )
        frame_kwargs = {"index": targets.index, "columns": targets.columns}
        beta = pd.DataFrame(eager.coef[:, -1], **frame_kwargs)
        residuals = pd.DataFrame(eager.resid_endpoint, **frame_kwargs)
        expected = {
            "get_beta": beta,
            "get_intercept": pd.DataFrame(eager.intercept, **frame_kwargs),
            "get_signal": beta.mul(factors[factor], axis=0),
            "get_r2": pd.DataFrame(1.0 - eager.ssr / eager.sst, **frame_kwargs),
            "get_partial_r2": pd.DataFrame(
                (reduced.ssr - eager.ssr) / reduced.ssr,
                **frame_kwargs,
            ),
            "get_residuals": residuals,
            "get_dof": pd.DataFrame(eager.n_eff - eager.df_eff, **frame_kwargs),
            "get_n_used": pd.DataFrame(eager.n_used, **frame_kwargs),
            "get_se": rolling_hac_se(
                targets,
                design,
                window=16,
                min_periods=11,
                expanding=False,
                n_lags=2,
                penalty=penalty,
            ),
        }
        expected["get_tstat"] = beta.div(expected["get_se"])
        for accessor, expected_values in expected.items():
            np.testing.assert_allclose(
                getattr(result, accessor)(factor), expected_values, rtol=0, atol=1e-12
            )
        for control_position, control in enumerate(controls):
            np.testing.assert_allclose(
                result.get_control_beta(factor, control),
                eager.coef[:, control_position],
                rtol=0,
                atol=1e-12,
            )

    controls_only = rolling_joint_solve(
        targets,
        controls,
        window=16,
        min_periods=11,
        expanding=False,
        penalty=controls_penalty,
        warn_singular=False,
    )
    np.testing.assert_allclose(
        result.get_factor_adjusted_returns(),
        controls_only.resid_endpoint,
        rtol=0,
        atol=1e-12,
    )


def test_full_and_partial_r2_contract_with_controls() -> None:
    rng = np.random.default_rng(31)
    index = pd.RangeIndex(40)
    control = rng.normal(size=40)
    factor = rng.normal(size=40)
    target = 4.0 * control + 0.5 * factor + rng.normal(scale=0.2, size=40)
    controls = pd.DataFrame({"control": control}, index=index)
    factors = pd.DataFrame({"factor": factor}, index=index)
    targets = pd.DataFrame({"asset": target}, index=index)
    result = RollingOLS(window=20, dtype="float64", mode="batched").fit_transform(
        factors, targets, controls
    )
    oracle = oracle_rolling(targets, factors, controls, window=20, min_periods=20, expanding=False)

    full_r2 = result.get_r2("factor")
    np.testing.assert_allclose(full_r2, oracle["r2"]["factor"], rtol=0, atol=1e-12)
    full_fit = oracle_fit_window(
        target[-20:],
        np.column_stack([control[-20:], factor[-20:]]),
        fit_intercept=True,
        weights=None,
        penalty=None,
    )
    reduced_fit = oracle_fit_window(
        target[-20:],
        control[-20:, None],
        fit_intercept=True,
        weights=None,
        penalty=None,
    )
    expected_partial = (reduced_fit.ssr - full_fit.ssr) / reduced_fit.ssr
    assert result.get_partial_r2("factor").iloc[-1, 0] == pytest.approx(expected_partial, abs=1e-12)
    assert abs(full_r2.iloc[-1, 0] - expected_partial) > 0.05


def test_r2_reconstructs_from_pattern_sufficient_statistics() -> None:
    targets, factors, controls = _small_panel(n_factors=2)
    result = RollingOLS(window=14, min_periods=10, dtype="float64", mode="batched").fit_transform(
        factors, targets, controls
    )
    actual = result.get_r2("f0").to_numpy()
    beta = result.get_beta("f0").to_numpy()

    for pattern in result._sufficient_statistics:
        assert not hasattr(pattern, "cross_products")
        matches = np.flatnonzero(pattern.factor_positions == 0)
        if matches.size == 0:
            continue
        local_factor = int(matches[0])
        endpoint_beta = beta[pattern.endpoint, pattern.target_positions]
        explained_ss = endpoint_beta**2 * pattern.denominators[local_factor]
        reconstructed = 1.0 - (pattern.reduced_ssr - explained_ss) / pattern.raw_sst
        np.testing.assert_allclose(
            actual[pattern.endpoint, pattern.target_positions],
            reconstructed,
            rtol=0,
            atol=1e-12,
        )


def test_ewma_adjusted_r2_uses_effective_not_raw_sample_size() -> None:
    targets, factors, controls = _small_panel(n_factors=1, n_controls=1)
    options = dict(window=18, min_periods=12, ewma_halflife=3, dtype="float64")
    adjusted = RollingOLS(adj_r2=True, **options, mode="batched").fit_transform(
        factors, targets, controls
    )
    unadjusted = RollingOLS(adj_r2=False, **options, mode="batched").fit_transform(
        factors, targets, controls
    )
    oracle = oracle_rolling(
        targets,
        factors,
        controls,
        window=18,
        min_periods=12,
        expanding=False,
        ewma_halflife=3,
    )
    np.testing.assert_allclose(adjusted.get_r2("f0"), oracle["adj_r2"]["f0"], rtol=0, atol=1e-12)

    n_used = unadjusted.get_n_used("f0").to_numpy()
    wrong_adjustment = (n_used - 1) / (n_used - 3)
    wrong = 1.0 - (1.0 - unadjusted.get_r2("f0").to_numpy()) * wrong_adjustment
    assert np.nanmax(np.abs(adjusted.get_r2("f0").to_numpy() - wrong)) > 1e-4


def test_effective_sample_size_identity_and_scale_invariance() -> None:
    weights = np.array([0.2, 0.5, 1.3, 2.0])
    expected = weights.sum() ** 2 / np.sum(weights**2)
    assert _effective_sample_size(weights) == pytest.approx(expected, abs=1e-15)
    assert _effective_sample_size(17.0 * weights) == pytest.approx(expected, abs=1e-15)
    assert _effective_sample_size(np.ones(7)) == 7.0


def test_factor_specific_missingness_matches_oracle() -> None:
    targets, factors, controls = _small_panel(n_factors=2)
    factors.iloc[8:13, 0] = np.nan
    factors.iloc[24:30, 1] = np.nan
    result = RollingOLS(window=16, min_periods=10, dtype="float64", mode="batched").fit_transform(
        factors, targets, controls
    )
    oracle = oracle_rolling(targets, factors, controls, window=16, min_periods=10, expanding=False)

    getter_by_quantity = {
        "beta": "get_beta",
        "intercept": "get_intercept",
        "residuals": "get_residuals",
        "r2": "get_r2",
        "dof": "get_dof",
        "n_used": "get_n_used",
    }
    for factor in factors:
        for quantity, getter in getter_by_quantity.items():
            np.testing.assert_allclose(
                getattr(result, getter)(factor),
                oracle[quantity][factor],
                rtol=0,
                atol=1e-12,
            )


def test_accessors_are_idempotent_order_independent_and_lru_bounded() -> None:
    targets, factors, controls = _small_panel()
    result = RollingOLS(
        window=14, min_periods=10, hac_lags=2, cache_size=1, mode="batched"
    ).fit_transform(factors, targets, controls)
    beta_before = result.get_beta("f0").copy()
    first_r2 = result.get_r2("f0")
    pd.testing.assert_frame_equal(first_r2, result.get_r2("f0"))
    pd.testing.assert_frame_equal(beta_before, result.get_beta("f0"))

    for factor in factors:
        result.get_residuals(factor)
        result.get_se(factor)
        assert len(result._residual_cache) <= 1
        assert len(result._se_cache) <= 1


def test_asset_subsetting_and_iterators() -> None:
    targets, factors, controls = _small_panel()
    result = RollingOLS(window=14, min_periods=10, hac_lags=2, mode="batched").fit_transform(
        factors, targets, controls
    )
    selected = ["a3", "a1"]
    for accessor in ("get_beta", "get_r2", "get_residuals", "get_se"):
        subset = getattr(result, accessor)("f0", assets=selected)
        if accessor == "get_r2":
            assert "f0" not in result._statistics_cache
        elif accessor == "get_residuals":
            assert "f0" not in result._residual_cache
        elif accessor == "get_se":
            assert "f0" not in result._se_cache
        full = getattr(result, accessor)("f0")
        pd.testing.assert_frame_equal(subset, full.loc[:, selected])
    adjusted_subset = result.get_factor_adjusted_returns(assets=selected)
    assert result._factor_adjusted_returns is None
    adjusted_full = result.get_factor_adjusted_returns()
    pd.testing.assert_frame_equal(adjusted_subset, adjusted_full.loc[:, selected])
    assert [factor for factor, _ in result.iter_beta()] == list(factors.columns)
    assert [factor for factor, _ in result.iter_se()] == list(factors.columns)


def test_estimate_memory_matches_persistent_small_result() -> None:
    targets, factors, controls = _small_panel(n_observations=60, n_targets=8, n_factors=4)
    model = RollingOLS(window=18, min_periods=12, dtype="float64", mode="batched")
    estimate = model.estimate_memory(targets, factors, controls)
    result = model.fit_transform(factors, targets, controls)
    primary_bytes = sum(
        frame.memory_usage(index=False, deep=False).sum()
        for mapping in (result._betas, result._intercepts, result._n_used)
        for frame in mapping.values()
    )
    statistic_bytes = sum(
        pattern.factor_positions.nbytes
        + pattern.target_positions.nbytes
        + pattern.denominators.nbytes
        + pattern.reduced_ssr.nbytes
        + pattern.raw_sst.nbytes
        for pattern in result._sufficient_statistics
    )
    retained_input_bytes = sum(
        frame.memory_usage(index=False, deep=False).sum() for frame in (targets, factors, controls)
    )
    actual = primary_bytes + statistic_bytes + retained_input_bytes
    assert int(estimate["total"]) == pytest.approx(actual, rel=0.2)


@pytest.mark.slow
def test_residual_iteration_has_bounded_live_cache() -> None:
    targets, factors, controls = _small_panel(
        n_observations=500,
        n_targets=120,
        n_factors=6,
        n_controls=2,
    )
    result = RollingOLS(window=60, cache_size=1, dtype="float64", mode="batched").fit_transform(
        factors, targets, controls
    )
    frame_bytes = len(targets) * targets.shape[1] * np.dtype(np.float64).itemsize
    tracemalloc.start()
    for _, residuals in ((factor, result.get_residuals(factor)) for factor in factors):
        assert residuals.shape == targets.shape
        assert len(result._residual_cache) <= 1
        del residuals
        gc.collect()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak <= (result.cache_size + 1) * frame_bytes + 12_000_000


@pytest.mark.slow
def test_se_iteration_streams_endpoints_with_bounded_memory() -> None:
    targets, factors, controls = _small_panel(
        n_observations=300,
        n_targets=60,
        n_factors=4,
        n_controls=2,
    )
    result = RollingOLS(
        window=40, hac_lags=3, cache_size=1, dtype="float64", mode="batched"
    ).fit_transform(factors, targets, controls)
    frame_bytes = len(targets) * targets.shape[1] * np.dtype(np.float64).itemsize
    tracemalloc.start()
    for _, standard_errors in result.iter_se():
        assert standard_errors.shape == targets.shape
        assert len(result._se_cache) <= 1
        del standard_errors
        gc.collect()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak <= (result.cache_size + 1) * frame_bytes + 12_000_000
