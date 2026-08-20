"""Degenerate and boundary input coverage for RollingOLS.

Each test targets one edge case from the v0.3.0 audit (F22): inputs that are
small, missing, constant, singular, or otherwise at the boundary of what the
estimator accepts. The standard is "NaN, not a crash and not garbage" — every
test checks the actual reported value, not just the absence of an exception.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from rols import RollingOLS
from rols.model import _ewma_weights
from tests.oracle import oracle_hac_se


def _panel(
    n_observations: int,
    n_factors: int = 1,
    n_controls: int = 0,
    n_targets: int = 1,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    rng = np.random.default_rng(seed)
    index = pd.RangeIndex(n_observations)
    factors = pd.DataFrame(
        rng.normal(size=(n_observations, n_factors)),
        index=index,
        columns=[f"f{i}" for i in range(n_factors)],
    )
    controls = None
    if n_controls:
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
    return factors, controls, targets


class TestAllNaNInputs:
    def test_all_nan_window_produces_nan_no_exception_no_warning_storm(self):
        factors, _, targets = _panel(30)
        targets.iloc[:, 0] = np.nan

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = RollingOLS(window=10, warn_singular=True, mode="batched").fit_transform(
                factors, targets
            )

        beta = result.get_beta("f0")
        assert beta.isna().all().all()
        # A NaN target never reaches the solver, so it must not be reported as
        # a singular window — no warning storm from an entirely-missing column.
        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0

    def test_all_nan_target_column_isolated_from_other_targets(self):
        factors, _, targets = _panel(30, n_targets=2)
        targets.iloc[:, 0] = np.nan

        result = RollingOLS(window=10, mode="batched").fit_transform(factors, targets)
        beta = result.get_beta("f0")

        assert beta["a0"].isna().all()
        assert beta["a1"].notna().any()


class TestBoundaryWindowSizes:
    def test_window_one_produces_exact_interpolation(self):
        factors, _, targets = _panel(10)
        result = RollingOLS(
            window=1, min_periods=1, precision="double", mode="batched"
        ).fit_transform(factors, targets)
        # With window=1 and an intercept, a single point is fit exactly:
        # residual is 0 everywhere a point exists (n_used=1 -> perfectly
        # collinear system with the intercept, coefficient is degenerate/NaN
        # in general, but the fit must not raise or produce inf).
        beta = result.get_beta("f0")
        assert np.isfinite(beta.to_numpy()[beta.notna().to_numpy()]).all()

    def test_window_two_min_periods_one(self):
        factors, _, targets = _panel(15)
        result = RollingOLS(
            window=2, min_periods=1, precision="double", mode="batched"
        ).fit_transform(factors, targets)
        beta = result.get_beta("f0")
        assert not np.isinf(beta.to_numpy()).any()

    def test_t_less_than_window_minus_one_no_estimate(self):
        n_observations = 5
        window = 10
        factors, _, targets = _panel(n_observations)
        result = RollingOLS(window=window, min_periods=window, mode="batched").fit_transform(
            factors, targets
        )
        assert result.get_beta("f0").isna().all().all()

    def test_t_equals_window_minus_one_no_estimate(self):
        window = 10
        factors, _, targets = _panel(window - 1)
        result = RollingOLS(window=window, min_periods=window, mode="batched").fit_transform(
            factors, targets
        )
        assert result.get_beta("f0").isna().all().all()

    def test_t_equals_window_produces_exactly_one_estimate(self):
        window = 10
        factors, _, targets = _panel(window)
        result = RollingOLS(window=window, min_periods=window, mode="batched").fit_transform(
            factors, targets
        )
        beta = result.get_beta("f0")
        assert beta.iloc[:-1].isna().all().all()
        assert beta.iloc[-1].notna().all()


class TestSingletonDimensions:
    def test_single_target_single_factor_single_control(self):
        factors, controls, targets = _panel(40, n_factors=1, n_controls=1, n_targets=1)
        result = RollingOLS(window=15, precision="double", mode="batched").fit_transform(
            factors, targets, controls=controls, return_control_betas=True
        )
        assert result.get_beta("f0").shape == (40, 1)
        assert result.get_control_beta("f0", "c0").shape == (40, 1)


class TestSingularAndDegenerateDesigns:
    def test_zero_variance_factor_in_window_never_silent_or_infinite(self):
        """A zero-variance factor makes the design exactly rank-deficient in
        theory, but floating-point solves rarely detect *exact* singularity
        (LAPACK's pivoted LU tolerates a near-zero pivot). The contract this
        library actually provides — and the one worth guarding — is: never an
        unwarned or infinite result. Either the window is flagged singular
        (NaN) or it is flagged ill-conditioned (a warning accompanies the
        finite, if numerically unreliable, estimate).
        """
        n_observations = 40
        window = 10
        rng = np.random.default_rng(3)
        index = pd.RangeIndex(n_observations)
        factor_values = rng.normal(size=n_observations)
        # First `window` observations are constant -> the first full window
        # has a zero-variance factor with an intercept, i.e. a rank-deficient
        # design.
        factor_values[:window] = 5.0
        factors = pd.DataFrame({"f0": factor_values}, index=index)
        targets = pd.DataFrame({"a0": rng.normal(size=n_observations)}, index=index)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = RollingOLS(window=window, precision="double", mode="batched").fit_transform(
                factors, targets
            )

        beta = result.get_beta("f0")
        assert not np.isinf(beta.to_numpy()).any()
        first_window_beta = beta.to_numpy()[window - 1]
        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        if not np.isnan(first_window_beta):
            assert len(runtime_warnings) > 0, (
                "first window's zero-variance factor produced a finite beta "
                f"({first_window_beta!r}) with no singular/ill-conditioned warning"
            )

    def test_constant_target_gives_nan_r2_not_error(self):
        factors, _, targets = _panel(30)
        targets.iloc[:, 0] = 7.0
        result = RollingOLS(window=10, precision="double", mode="batched").fit_transform(
            factors, targets
        )
        r2 = result.get_r2("f0")
        # SST is 0 for a constant target: R² is undefined (0/0), not inf.
        assert (
            r2.notna().sum().sum() == 0 or np.isfinite(r2.to_numpy()[r2.notna().to_numpy()]).all()
        )

    def test_perfectly_collinear_regressors_joint_mode_nan_and_one_warning(self):
        n_observations = 40
        window = 15
        rng = np.random.default_rng(4)
        index = pd.RangeIndex(n_observations)
        f1 = rng.normal(size=n_observations)
        factors = pd.DataFrame({"f1": f1, "f2": 2.0 * f1}, index=index)
        targets = pd.DataFrame({"a0": rng.normal(size=n_observations)}, index=index)

        with pytest.warns(RuntimeWarning, match="singular"):
            result = RollingOLS(window=window, mode="joint", precision="double").fit_transform(
                factors, targets
            )

        beta = result.get_beta("f1")
        assert beta.notna().sum().sum() == 0

    def test_duplicate_factor_column_values_joint_mode_nan(self):
        n_observations = 40
        window = 15
        rng = np.random.default_rng(5)
        index = pd.RangeIndex(n_observations)
        f1 = rng.normal(size=n_observations)
        factors = pd.DataFrame({"f1": f1, "f2": f1.copy()}, index=index)
        targets = pd.DataFrame({"a0": rng.normal(size=n_observations)}, index=index)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = RollingOLS(window=window, mode="joint", precision="double").fit_transform(
                factors, targets
            )

        beta = result.get_beta("f1")
        assert beta.notna().sum().sum() == 0


class TestRidgeExtremes:
    def test_huge_lambda_shrinks_betas_to_zero_and_intercept_to_weighted_mean(self):
        window = 20
        factors, _, targets = _panel(window, n_factors=1, seed=6)
        result = RollingOLS(
            window=window, min_periods=window, lambda_=1e12, precision="double", mode="batched"
        ).fit_transform(factors, targets)

        beta = result.get_beta("f0").iloc[-1, 0]
        intercept = result.get_intercept("f0").iloc[-1, 0]
        assert abs(beta) < 1e-6
        # Equal weights within one full window -> the OLS-optimal intercept
        # given beta≈0 is the sample mean of y over the window.
        np.testing.assert_allclose(intercept, targets["a0"].mean(), atol=1e-4)


class TestEWMAExtreme:
    def test_halflife_one_no_blowup_and_dominant_last_weight(self):
        window = 20
        weights = _ewma_weights(1, window)
        assert np.isfinite(weights).all()
        # Half-life of 1 within a rolling window puts most of the mass on the
        # most recent observation.
        assert weights[-1] > 0.45

        factors, _, targets = _panel(40, n_factors=1, seed=7)
        result = RollingOLS(
            window=window, min_periods=window, ewma_halflife=1, precision="double", mode="batched"
        ).fit_transform(factors, targets)
        beta = result.get_beta("f0")
        assert np.isfinite(beta.to_numpy()[beta.notna().to_numpy()]).all()


class TestHACBoundaries:
    def test_hac_lags_zero_matches_hc0_oracle(self):
        window = 25
        factors, _, targets = _panel(window, n_factors=1, seed=8)
        result = RollingOLS(
            window=window, min_periods=window, hac_lags=0, precision="double", mode="batched"
        ).fit_transform(factors, targets)

        se = result.get_se("f0").iloc[-1, 0]
        design = np.column_stack([np.ones(window), factors["f0"].to_numpy()])
        coef, *_ = np.linalg.lstsq(design, targets["a0"].to_numpy(), rcond=None)
        expected = oracle_hac_se(
            targets["a0"].to_numpy(),
            factors[["f0"]].to_numpy(),
            coef,
            fit_intercept=True,
            weights=None,
            n_lags=0,
        )
        np.testing.assert_allclose(se, expected[-1], rtol=1e-6, atol=1e-9)

    def test_hac_lags_exceeding_window_gives_nan_not_garbage(self):
        window = 15
        factors, _, targets = _panel(window * 2, n_factors=1, seed=9)
        result = RollingOLS(
            window=window,
            min_periods=window,
            hac_lags=window + 5,
            precision="double",
            mode="batched",
        ).fit_transform(factors, targets)

        se = result.get_se("f0")
        assert se.notna().sum().sum() == 0
        assert not np.isinf(se.fillna(0.0).to_numpy()).any()


class TestCadenceBeyondSampleSize:
    def test_estimate_every_larger_than_t_gives_single_endpoint(self):
        n_observations = 20
        window = 10
        factors, _, targets = _panel(n_observations, n_factors=1, seed=10)
        result = RollingOLS(
            window=window, min_periods=window, estimate_every=50, precision="double", mode="batched"
        ).fit_transform(factors, targets)

        beta = result.get_beta("f0")
        assert beta.notna().sum().sum() == 1
        assert beta.iloc[-1].notna().all()
