"""Independent checks of the scalar statistical oracle."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import assert_matches_oracle
from tests.oracle import oracle_fit_window, oracle_hac_se, oracle_rolling


def _ridge_penalty(n_slopes: int, lambda_: float, fit_intercept: bool = True) -> np.ndarray:
    penalty = np.eye(n_slopes + int(fit_intercept)) * lambda_
    if fit_intercept:
        penalty[0, 0] = 0.0
    return penalty


def test_fit_window_matches_numpy_lstsq() -> None:
    rng = np.random.default_rng(1)
    regressors = rng.normal(size=(40, 3))
    target = 1.5 + regressors @ np.array([0.25, -0.5, 2.0]) + rng.normal(size=40)

    fit = oracle_fit_window(target, regressors, True, None, None)
    expected = np.linalg.lstsq(np.column_stack([np.ones(40), regressors]), target, rcond=None)[0]

    np.testing.assert_allclose(
        np.concatenate([[fit.intercept], fit.coef]), expected, rtol=0, atol=1e-12
    )


def test_known_intercept_and_slope_closed_form() -> None:
    factor = np.linspace(-2.0, 3.0, 20)[:, None]
    target = 3.0 + 2.0 * factor[:, 0]

    fit = oracle_fit_window(target, factor, True, None, None)

    assert fit.intercept == pytest.approx(3.0, abs=1e-12)
    assert fit.coef[0] == pytest.approx(2.0, abs=1e-12)
    np.testing.assert_allclose(fit.resid, 0.0, atol=1e-12)
    assert 1 - fit.ssr / fit.sst == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("n_observations", [20, 80])
def test_ridge_has_same_closed_form_across_window_lengths(n_observations: int) -> None:
    rng = np.random.default_rng(n_observations)
    raw = rng.normal(size=(n_observations, 2))
    centered = raw - raw.mean(axis=0)
    orthogonal, _ = np.linalg.qr(centered)
    regressors = orthogonal * np.sqrt(n_observations)
    true_coef = np.array([2.0, -1.0])
    target = regressors @ true_coef
    lambda_ = 0.25

    fit = oracle_fit_window(
        target,
        regressors,
        True,
        None,
        _ridge_penalty(2, lambda_),
    )

    np.testing.assert_allclose(fit.coef, true_coef / (1 + lambda_), atol=1e-12)


def test_uniform_weights_reproduce_unweighted_ridge() -> None:
    rng = np.random.default_rng(4)
    regressors = rng.normal(loc=2.0, scale=3.0, size=(35, 2))
    target = 4.0 + regressors @ np.array([1.0, -0.5]) + rng.normal(size=35)
    penalty = _ridge_penalty(2, 0.7)

    unweighted = oracle_fit_window(target, regressors, True, None, penalty)
    weighted = oracle_fit_window(target, regressors, True, np.ones(35), penalty)

    np.testing.assert_allclose(weighted.coef, unweighted.coef, atol=1e-12)
    assert weighted.intercept == pytest.approx(unweighted.intercept, abs=1e-12)
    np.testing.assert_allclose(weighted.resid, unweighted.resid, atol=1e-12)


def test_complete_case_matches_explicit_row_deletion() -> None:
    rng = np.random.default_rng(5)
    regressors = rng.normal(size=(25, 2))
    target = 1.0 + regressors @ np.array([0.5, 2.0]) + rng.normal(size=25)
    regressors[7, 0] = np.nan
    target[13] = np.nan
    complete = np.isfinite(target) & np.isfinite(regressors).all(axis=1)

    with_nans = oracle_fit_window(target, regressors, True, None, None)
    deleted = oracle_fit_window(target[complete], regressors[complete], True, None, None)

    np.testing.assert_allclose(with_nans.coef, deleted.coef, atol=1e-12)
    assert with_nans.intercept == pytest.approx(deleted.intercept, abs=1e-12)
    np.testing.assert_allclose(with_nans.resid[complete], deleted.resid, atol=1e-12)
    assert np.isnan(with_nans.resid[~complete]).all()


@pytest.mark.parametrize("expanding", [False, True])
def test_rolling_driver_with_controls_recovers_exact_model(expanding: bool) -> None:
    index = pd.RangeIndex(30)
    factor = np.linspace(-1.0, 2.0, len(index))
    control = np.sin(np.arange(len(index)))
    targets = pd.DataFrame({"asset": 3.0 + 2.0 * factor + 0.5 * control}, index=index)
    factors = pd.DataFrame({"factor": factor}, index=index)
    controls = pd.DataFrame({"control": control}, index=index)

    result = oracle_rolling(
        targets,
        factors,
        controls,
        window=12,
        min_periods=8,
        expanding=expanding,
    )

    assert result["beta"]["factor"].iloc[-1, 0] == pytest.approx(2.0, abs=1e-11)
    assert result["control_beta"]["factor"]["control"].iloc[-1, 0] == pytest.approx(0.5, abs=1e-11)
    assert result["intercept"]["factor"].iloc[-1, 0] == pytest.approx(3.0, abs=1e-11)
    assert result["residuals"]["factor"].iloc[-1, 0] == pytest.approx(0.0, abs=1e-11)
    assert result["r2"]["factor"].iloc[-1, 0] == pytest.approx(1.0, abs=1e-12)


def test_rolling_driver_covers_joint_ridge_ewma_no_intercept_and_nans() -> None:
    rng = np.random.default_rng(6)
    index = pd.RangeIndex(24)
    factors = pd.DataFrame(rng.normal(size=(24, 2)), index=index, columns=["f1", "f2"])
    targets = pd.DataFrame(
        {"asset": factors["f1"] - 0.5 * factors["f2"] + rng.normal(scale=0.1, size=24)},
        index=index,
    )
    targets.iloc[5, 0] = np.nan
    factors.iloc[9, 1] = np.nan

    result = oracle_rolling(
        targets,
        factors,
        controls=None,
        window=10,
        min_periods=6,
        expanding=False,
        fit_intercept=False,
        lambda_=0.2,
        ewma_halflife=4,
        mode="joint",
    )

    assert np.isfinite(result["beta"]["f1"].iloc[-1, 0])
    assert np.isfinite(result["beta"]["f2"].iloc[-1, 0])
    assert result["intercept"]["f1"].iloc[-1, 0] == 0.0
    assert result["n_used"]["f1"].iloc[9, 0] == 8
    assert result["n_eff"]["f1"].iloc[-1, 0] < result["n_used"]["f1"].iloc[-1, 0]
    pd.testing.assert_frame_equal(result["r2"]["f1"], result["r2"]["f2"])


def test_penalize_controls_false_leaves_control_unshrunk() -> None:
    index = pd.RangeIndex(30)
    control = np.linspace(-2.0, 2.0, len(index))
    factor = np.sin(np.arange(len(index), dtype=np.float64))
    nuisance_design = np.column_stack([np.ones(len(index)), control])
    nuisance_coef = np.linalg.lstsq(nuisance_design, factor, rcond=None)[0]
    factor = factor - nuisance_design @ nuisance_coef
    targets = pd.DataFrame({"asset": 4.0 * control + 2.0 * factor}, index=index)
    factors = pd.DataFrame({"factor": factor}, index=index)
    controls = pd.DataFrame({"control": control}, index=index)

    result = oracle_rolling(
        targets,
        factors,
        controls,
        window=30,
        min_periods=30,
        expanding=False,
        lambda_=1.0,
        penalize_controls=False,
    )

    control_beta = result["control_beta"]["factor"]["control"].iloc[-1, 0]
    assert control_beta == pytest.approx(4.0, abs=1e-12)


def test_hac_lag_zero_matches_manual_weighted_sandwich() -> None:
    rng = np.random.default_rng(7)
    regressors = rng.normal(size=(40, 1))
    target = 1.0 + 0.75 * regressors[:, 0] + rng.normal(size=40)
    fit = oracle_fit_window(target, regressors, True, None, None)
    coefficients = np.array([fit.intercept, fit.coef[0]])

    actual = oracle_hac_se(target, regressors, coefficients, True, None, n_lags=0)

    design = np.column_stack([np.ones(40), regressors])
    weights = np.full(40, 1 / 40)
    scores = weights[:, None] * design * fit.resid[:, None]
    bread_inverse = np.linalg.inv(design.T @ (weights[:, None] * design))
    covariance = 40 / (40 - 2) * bread_inverse @ (scores.T @ scores) @ bread_inverse
    np.testing.assert_allclose(actual, np.sqrt(np.diag(covariance)), atol=1e-12)


def test_matches_statsmodels_when_available() -> None:
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(8)
    regressors = rng.normal(size=(60, 2))
    target = 2.0 + regressors @ np.array([0.5, -1.5]) + rng.normal(size=60)
    fit = oracle_fit_window(target, regressors, True, None, None)
    design = sm.add_constant(regressors)
    expected = sm.OLS(target, design).fit()

    np.testing.assert_allclose(
        np.concatenate([[fit.intercept], fit.coef]), expected.params, atol=1e-12
    )
    r2 = 1 - fit.ssr / fit.sst
    adj_r2 = 1 - (1 - r2) * (fit.n_eff - 1) / (fit.n_eff - regressors.shape[1] - 1)
    assert r2 == pytest.approx(expected.rsquared, abs=1e-12)
    assert adj_r2 == pytest.approx(expected.rsquared_adj, abs=1e-12)

    actual_se = oracle_hac_se(target, regressors, expected.params, True, None, n_lags=3)
    expected_hac = expected.get_robustcov_results(cov_type="HAC", maxlags=3, use_correction=True)
    np.testing.assert_allclose(actual_se, expected_hac.bse, atol=1e-12)


def test_differential_helper_rejects_nan_number_mismatch() -> None:
    frame = pd.DataFrame({"asset": [np.nan]})
    quantities = ("beta", "intercept", "residuals", "r2", "adj_r2", "dof", "n_used", "n_eff")
    expected = {quantity: {"factor": frame.copy()} for quantity in quantities}
    actual = {quantity: {"factor": frame.copy()} for quantity in quantities}
    expected["control_beta"] = {}
    actual["control_beta"] = {}
    actual["beta"]["factor"].iloc[0, 0] = 1.0

    with pytest.raises(AssertionError, match="actual=1.0, expected=nan"):
        assert_matches_oracle(actual, expected)
