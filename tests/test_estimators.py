"""Tests for low-level rolling estimators."""

import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import rols.estimators as est
from rols import RollingOLS
from rols.estimators import (
    _solve_batch,
    hac_se,
    rolling_joint_solve,
    rolling_residualize,
)
from rols.model import _ewma_weights


def test_removed_gram_schmidt_estimator_is_absent():
    assert not hasattr(est, "rolling_gram_schmidt")


class TestRollingJointSolve:
    """Tests for the authoritative current-window joint solver."""

    def test_exact_intercept_model_is_coherent(self):
        index = pd.RangeIndex(30)
        factor = pd.DataFrame({"factor": np.linspace(-2.0, 3.0, len(index))}, index=index)
        targets = pd.DataFrame({"asset": 3.0 + 2.0 * factor["factor"]}, index=index)

        fit = rolling_joint_solve(
            targets,
            factor,
            window=12,
            min_periods=8,
            expanding=False,
            fit_intercept=True,
        )

        np.testing.assert_allclose(fit.intercept[7:, 0], 3.0, atol=1e-12)
        np.testing.assert_allclose(fit.coef[7:, 0, 0], 2.0, atol=1e-12)
        np.testing.assert_allclose(fit.resid_endpoint[7:, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(1 - fit.ssr[7:, 0] / fit.sst[7:, 0], 1.0, atol=1e-12)

    def test_endpoint_identity_holds_for_every_reported_fit(self):
        rng = np.random.default_rng(12)
        index = pd.RangeIndex(50)
        design = pd.DataFrame(
            rng.normal(size=(50, 3)), index=index, columns=["control_1", "control_2", "factor"]
        )
        targets = pd.DataFrame(
            2.0 + design.to_numpy() @ rng.normal(size=(3, 2)) + rng.normal(size=(50, 2)),
            index=index,
            columns=["asset_1", "asset_2"],
        )

        fit = rolling_joint_solve(
            targets,
            design,
            window=15,
            min_periods=8,
            expanding=False,
            fit_intercept=True,
        )
        fitted_endpoint = fit.intercept + np.einsum("tp,tpn->tn", design.to_numpy(), fit.coef)
        reconstructed = fitted_endpoint + fit.resid_endpoint
        valid = np.isfinite(reconstructed)

        np.testing.assert_allclose(
            reconstructed[valid], targets.to_numpy()[valid], rtol=0, atol=1e-10
        )


class TestJointSolverConditioning:
    """The joint solver factorizes the design and reports unstable windows."""

    @staticmethod
    def _ill_conditioned_inputs(
        n_observations: int = 40,
    ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        rng = np.random.default_rng(0)
        left, _ = np.linalg.qr(rng.normal(size=(n_observations, 2)))
        right, _ = np.linalg.qr(rng.normal(size=(2, 2)))
        design_values = left @ np.diag([1.0, 1e-8]) @ right.T
        expected = np.array([1.25, -0.75])
        targets = pd.DataFrame({"asset": design_values @ expected})
        design = pd.DataFrame(design_values, columns=["x1", "x2"])
        return targets, design, expected

    def test_qr_recovers_ill_conditioned_coefficients(self):
        targets, design, expected = self._ill_conditioned_inputs()

        with pytest.warns(RuntimeWarning, match=r"cond\(X'X\)"):
            fit = rolling_joint_solve(
                targets,
                design,
                window=40,
                min_periods=40,
                expanding=False,
                fit_intercept=False,
            )

        actual = fit.coef[-1, :, 0]
        normal_equations = np.linalg.solve(
            design.to_numpy().T @ design.to_numpy(),
            design.to_numpy().T @ targets.to_numpy(),
        )[:, 0]
        assert np.linalg.cond(design) == pytest.approx(1e8, rel=1e-8)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
        assert np.max(np.abs(normal_equations - expected)) > 1e-2

    def test_diagnostic_is_aggregated_across_asset_chunks(self):
        targets, design, _ = self._ill_conditioned_inputs()
        targets = pd.concat(
            [targets.rename(columns={"asset": f"asset_{i}"}) for i in range(3)], axis=1
        )

        model = RollingOLS(
            window=40,
            min_periods=40,
            fit_intercept=False,
            dtype="float64",
            asset_chunk_size=1,
            cond_warn_threshold=1e10,
        )
        with pytest.warns(RuntimeWarning, match="ill-conditioned") as captured:
            model.fit_transform(design[["x1"]], targets, controls=design[["x2"]])

        assert len(captured) == 1
        assert "3 window(s)" in str(captured[0].message)

    def test_well_conditioned_input_is_silent(self):
        rng = np.random.default_rng(34)
        design = pd.DataFrame(rng.normal(size=(40, 2)), columns=["x1", "x2"])
        targets = pd.DataFrame({"asset": 1.5 * design["x1"] - 0.5 * design["x2"]})

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            rolling_joint_solve(
                targets,
                design,
                window=20,
                min_periods=20,
                expanding=False,
                fit_intercept=False,
            )

    def test_warning_can_be_suppressed(self, recwarn):
        targets, design, _ = self._ill_conditioned_inputs()

        rolling_joint_solve(
            targets,
            design,
            window=40,
            min_periods=40,
            expanding=False,
            fit_intercept=False,
            warn_singular=False,
        )

        assert len(recwarn) == 0

    def test_rank_deficient_window_is_nan_and_warned(self):
        rng = np.random.default_rng(35)
        column = rng.normal(size=30)
        design = pd.DataFrame({"x1": column, "x2": column})
        targets = pd.DataFrame({"asset": rng.normal(size=30)})

        with pytest.warns(RuntimeWarning, match="1 singular window") as captured:
            fit = rolling_joint_solve(
                targets,
                design,
                window=30,
                min_periods=30,
                expanding=False,
                fit_intercept=False,
            )

        assert len(captured) == 1
        assert np.isnan(fit.coef[-1]).all()
        assert not np.isinf(fit.coef).any()

    def test_qr_agrees_with_normal_equations_when_well_conditioned(self):
        rng = np.random.default_rng(36)
        design = pd.DataFrame(rng.normal(size=(50, 3)), columns=["x1", "x2", "x3"])
        targets = pd.DataFrame(rng.normal(size=(50, 2)), columns=["a1", "a2"])

        fit = rolling_joint_solve(
            targets,
            design,
            window=50,
            min_periods=50,
            expanding=False,
            fit_intercept=False,
        )
        normal_equations = np.linalg.solve(
            design.to_numpy().T @ design.to_numpy(),
            design.to_numpy().T @ targets.to_numpy(),
        )

        np.testing.assert_allclose(fit.coef[-1], normal_equations, rtol=0, atol=1e-10)


class TestRollingResidualize:
    """Tests for rolling_residualize function."""

    def test_basic_ols_rolling(self):
        """Test basic OLS residualization with rolling window."""
        np.random.seed(42)
        T = 100
        y = pd.DataFrame(np.random.randn(T, 2), columns=["y1", "y2"])
        X = pd.DataFrame(np.random.randn(T, 2), columns=["x1", "x2"])

        result = rolling_residualize(
            y=y, X=X, window=10, min_periods=10, expanding=False, ridge_lambda=0.0
        )

        assert result.shape == y.shape
        assert list(result.columns) == list(y.columns)
        assert isinstance(result, pd.DataFrame)

    def test_ols_expanding(self):
        """Test OLS with expanding window."""
        np.random.seed(42)
        T = 50
        y = pd.DataFrame(np.random.randn(T, 1), columns=["y"])
        X = pd.DataFrame(np.random.randn(T, 1), columns=["x"])

        min_periods = 5
        result = rolling_residualize(
            y=y, X=X, window=10, min_periods=min_periods, expanding=True, ridge_lambda=0.0
        )

        assert result.shape == y.shape
        # First few values should be NaN (before min_periods)
        assert result.iloc[:4].isna().all().all()
        # After min_periods, should have values
        assert result.iloc[min_periods - 1 :].notna().any().any()

    def test_ridge_regression(self):
        """Test Ridge regularization."""
        np.random.seed(42)
        T = 100
        y = pd.DataFrame(np.random.randn(T, 1), columns=["y"])
        X = pd.DataFrame(np.random.randn(T, 2), columns=["x1", "x2"])

        resid_ols = rolling_residualize(
            y=y, X=X, window=20, min_periods=20, expanding=False, ridge_lambda=0.0
        )
        resid_ridge = rolling_residualize(
            y=y, X=X, window=20, min_periods=20, expanding=False, ridge_lambda=1.0
        )

        # Ridge should produce different (generally smaller) residuals
        assert not resid_ols.equals(resid_ridge)

    def test_nan_handling_y(self):
        """Test NaN handling in target variable."""
        np.random.seed(42)
        T = 50
        y = pd.DataFrame(np.random.randn(T, 2), columns=["y1", "y2"])
        y.iloc[10:15, 0] = np.nan  # NaNs in first column
        X = pd.DataFrame(np.random.randn(T, 1), columns=["x"])

        result = rolling_residualize(
            y=y, X=X, window=10, min_periods=5, expanding=False, ridge_lambda=0.0
        )

        # Result should handle NaNs without crashing
        assert result.shape == y.shape
        # NaN rows in y should result in NaN residuals
        assert result.iloc[10:15, 0].isna().all()

    def test_nan_handling_X(self):
        """Test NaN handling in regressor matrix."""
        np.random.seed(42)
        T = 50
        y = pd.DataFrame(np.random.randn(T, 1), columns=["y"])
        X = pd.DataFrame(np.random.randn(T, 2), columns=["x1", "x2"])
        X.iloc[10:15, 0] = np.nan  # NaNs in regressor

        result = rolling_residualize(
            y=y, X=X, window=10, min_periods=5, expanding=False, ridge_lambda=0.0
        )

        assert result.shape == y.shape
        # Windows containing NaN in X should have some NaN residuals
        # but the behavior might be more nuanced than full NaNification
        assert result.isna().any().any()

    def test_min_periods_early_windows(self):
        """Test min_periods parameter for early windows."""
        np.random.seed(42)
        T = 50
        y = pd.DataFrame(np.random.randn(T, 1), columns=["y"])
        X = pd.DataFrame(np.random.randn(T, 1), columns=["x"])

        result = rolling_residualize(
            y=y, X=X, window=20, min_periods=5, expanding=False, ridge_lambda=0.0
        )

        # First 4 observations should be NaN (min_periods=5)
        assert result.iloc[:4].isna().all().all()
        # From position 4 onwards, should have some non-NaN values
        assert result.iloc[4:].notna().any().any()

    def test_single_regressor(self):
        """Test with single regressor."""
        np.random.seed(42)
        T = 100
        y = pd.DataFrame(np.random.randn(T, 1), columns=["y"])
        X = pd.DataFrame(np.random.randn(T, 1), columns=["x"])

        result = rolling_residualize(
            y=y, X=X, window=20, min_periods=20, expanding=False, ridge_lambda=0.0
        )

        assert result.shape == (T, 1)

    def test_multiple_targets(self):
        """Test with multiple target variables."""
        np.random.seed(42)
        T = 100
        y = pd.DataFrame(np.random.randn(T, 5), columns=[f"y{i}" for i in range(5)])
        X = pd.DataFrame(np.random.randn(T, 2), columns=["x1", "x2"])

        result = rolling_residualize(
            y=y, X=X, window=20, min_periods=20, expanding=False, ridge_lambda=0.0
        )

        assert result.shape == (T, 5)
        assert list(result.columns) == list(y.columns)

    def test_window_size_equals_series_length(self):
        """Test when window size equals series length."""
        np.random.seed(42)
        T = 20
        y = pd.DataFrame(np.random.randn(T, 1), columns=["y"])
        X = pd.DataFrame(np.random.randn(T, 1), columns=["x"])

        result = rolling_residualize(
            y=y, X=X, window=T, min_periods=T, expanding=False, ridge_lambda=0.0
        )

        # Only last observation should be non-NaN
        assert result.iloc[-1].notna().all()

    def test_float32_inputs_finite(self):
        """float32 inputs must not raise and must produce finite residuals (issue #10)."""
        np.random.seed(42)
        T = 100
        y = pd.DataFrame(np.random.randn(T, 3).astype(np.float32), columns=["y1", "y2", "y3"])
        X = pd.DataFrame(np.random.randn(T, 2).astype(np.float32), columns=["x1", "x2"])

        result = rolling_residualize(
            y=y, X=X, window=20, min_periods=20, expanding=False, ridge_lambda=0.0
        )

        assert result.shape == y.shape
        valid = result.to_numpy()[~np.isnan(result.to_numpy())]
        assert len(valid) > 0
        assert np.isfinite(valid).all()

    def test_float32_matches_float64_baseline(self):
        """float32 inputs give results numerically close to the float64 baseline."""
        np.random.seed(0)
        T = 120
        y64 = pd.DataFrame(np.random.randn(T, 2), columns=["y1", "y2"])
        X64 = pd.DataFrame(np.random.randn(T, 2), columns=["x1", "x2"])

        result64 = rolling_residualize(
            y=y64, X=X64, window=20, min_periods=20, expanding=False, ridge_lambda=0.0
        )
        result32 = rolling_residualize(
            y=y64.astype(np.float32),
            X=X64.astype(np.float32),
            window=20,
            min_periods=20,
            expanding=False,
            ridge_lambda=0.0,
        )

        # Compare where both are defined — within float32 tolerance.
        np.testing.assert_allclose(
            result32.to_numpy(), result64.to_numpy(), atol=1e-4, equal_nan=True
        )


class TestVectorizedNaNRobustPath:
    """Tests for the intermediate vectorized NaN-robust path (issue #2).

    Path selection:
      1. no NaNs              -> fast vectorized path
      2. NaNs only in y       -> intermediate path (O(N) loop, T vectorized)
      3. NaNs in X            -> per-column fallback (_residualize_single)

    Performance: the intermediate path replaces the O(T * N) Python loop of the
    per-column fallback with an O(N) loop (T vectorized inside each iteration),
    giving a large speedup on sparse panels (e.g. ~10-50x at MSCI-World scale,
    ~2300 assets, where NaNs from constituent entry/exit are always present).
    """

    def _reference_per_column(self, y, X, window, min_periods, ridge_lambda=0.0):
        """Compute residuals via the per-column loop (_residualize_single)."""
        y_np = y.to_numpy(np.float64)
        X_np = X.to_numpy(np.float64)
        T, N = y_np.shape
        k = X_np.shape[1]
        ridge_term = ridge_lambda * np.eye(k)
        x_row_valid = ~np.isnan(X_np).any(axis=1)
        ref = np.full((T, N), np.nan)
        for j in range(N):
            ref[:, j], _ = est._residualize_single(
                y_np[:, j], X_np, T, window, min_periods, ridge_term, x_row_valid
            )
        return ref

    def test_matches_per_column_loop(self):
        """Intermediate path must match the per-column loop exactly."""
        np.random.seed(3)
        T, N, k = 90, 7, 2
        y = pd.DataFrame(np.random.randn(T, N), columns=[f"y{i}" for i in range(N)])
        X = pd.DataFrame(np.random.randn(T, k), columns=["x1", "x2"])
        # Scattered NaNs in y only (X stays clean) -> hits the intermediate path.
        y.iloc[10:20, 0] = np.nan
        y.iloc[30, 2] = np.nan
        y.iloc[5:8, 4] = np.nan
        y.iloc[40:45, 6] = np.nan

        result = rolling_residualize(
            y=y, X=X, window=20, min_periods=20, expanding=False, ridge_lambda=0.0
        )
        ref = self._reference_per_column(y, X, window=20, min_periods=20)
        np.testing.assert_allclose(result.to_numpy(), ref, equal_nan=True, atol=1e-12)

    def test_matches_per_column_loop_min_periods_lt_window(self):
        """Match the per-column loop including the early-window branch."""
        np.random.seed(4)
        T, N, k = 70, 5, 2
        y = pd.DataFrame(np.random.randn(T, N), columns=[f"y{i}" for i in range(N)])
        X = pd.DataFrame(np.random.randn(T, k), columns=["x1", "x2"])
        y.iloc[12:18, 1] = np.nan
        y.iloc[3, 3] = np.nan

        result = rolling_residualize(
            y=y, X=X, window=20, min_periods=10, expanding=False, ridge_lambda=0.0
        )
        ref = self._reference_per_column(y, X, window=20, min_periods=10)
        np.testing.assert_allclose(result.to_numpy(), ref, equal_nan=True, atol=1e-12)

    def test_matches_per_column_loop_with_ridge(self):
        """Match the per-column loop when Ridge regularization is active."""
        np.random.seed(5)
        T, N, k = 80, 4, 2
        y = pd.DataFrame(np.random.randn(T, N), columns=[f"y{i}" for i in range(N)])
        X = pd.DataFrame(np.random.randn(T, k), columns=["x1", "x2"])
        y.iloc[15:25, 2] = np.nan

        result = rolling_residualize(
            y=y, X=X, window=20, min_periods=20, expanding=False, ridge_lambda=0.5
        )
        ref = self._reference_per_column(y, X, window=20, min_periods=20, ridge_lambda=0.5)
        np.testing.assert_allclose(result.to_numpy(), ref, equal_nan=True, atol=1e-12)

    def test_fast_path_used_when_no_nans(self):
        """No NaNs anywhere -> fast path (per-column loop never invoked)."""
        np.random.seed(6)
        T, N = 60, 4
        y = pd.DataFrame(np.random.randn(T, N), columns=[f"y{i}" for i in range(N)])
        X = pd.DataFrame(np.random.randn(T, 2), columns=["x1", "x2"])
        with patch.object(est, "_residualize_single", wraps=est._residualize_single) as m:
            rolling_residualize(y=y, X=X, window=15, min_periods=15, expanding=False)
        assert m.call_count == 0

    def test_intermediate_path_used_when_nans_only_in_y(self):
        """NaNs only in y -> intermediate path (per-column loop never invoked)."""
        np.random.seed(7)
        T, N = 60, 4
        y = pd.DataFrame(np.random.randn(T, N), columns=[f"y{i}" for i in range(N)])
        y.iloc[10:14, 1] = np.nan
        X = pd.DataFrame(np.random.randn(T, 2), columns=["x1", "x2"])
        with patch.object(est, "_residualize_single", wraps=est._residualize_single) as m:
            rolling_residualize(y=y, X=X, window=15, min_periods=15, expanding=False)
        assert m.call_count == 0

    def test_fallback_used_when_nans_in_X(self):
        """NaNs in X -> per-column fallback (one call per asset)."""
        np.random.seed(8)
        T, N = 60, 4
        y = pd.DataFrame(np.random.randn(T, N), columns=[f"y{i}" for i in range(N)])
        X = pd.DataFrame(np.random.randn(T, 2), columns=["x1", "x2"])
        X.iloc[10:14, 0] = np.nan
        with patch.object(est, "_residualize_single", wraps=est._residualize_single) as m:
            rolling_residualize(y=y, X=X, window=15, min_periods=15, expanding=False)
        assert m.call_count == N

    def test_nan_isolated_per_asset(self):
        """A NaN in one asset's y must not contaminate other assets."""
        np.random.seed(9)
        T, N = 80, 4
        y = pd.DataFrame(np.random.randn(T, N), columns=[f"y{i}" for i in range(N)])
        X = pd.DataFrame(np.random.randn(T, 2), columns=["x1", "x2"])

        clean = rolling_residualize(y=y, X=X, window=20, min_periods=20, expanding=False)
        contaminated = y.copy()
        contaminated.iloc[50, 1] = np.nan
        contam = rolling_residualize(
            y=contaminated, X=X, window=20, min_periods=20, expanding=False
        )

        # Untouched columns are bit-identical to the all-clean run.
        for col in ["y0", "y2", "y3"]:
            pd.testing.assert_series_equal(clean[col], contam[col])

    def test_prediction_point_nan_yields_nan(self):
        """When y_j is NaN at the prediction point, that residual is NaN."""
        np.random.seed(10)
        T, N = 60, 2
        y = pd.DataFrame(np.random.randn(T, N), columns=["y0", "y1"])
        X = pd.DataFrame(np.random.randn(T, 2), columns=["x1", "x2"])
        y.iloc[40, 0] = np.nan
        result = rolling_residualize(y=y, X=X, window=20, min_periods=20, expanding=False)
        # The prediction point itself is NaN.
        assert np.isnan(result.iloc[40, 0])
        # A window ending before the NaN row (t=39, rows [20:40)) is unaffected.
        assert not np.isnan(result.iloc[39, 0])


class TestEWMAWeights:
    """Tests for the EWMA weight vector helper."""

    def test_weights_sum_to_one(self):
        w = _ewma_weights(halflife=10, window=50)
        assert w.shape == (50,)
        assert w.sum() == pytest.approx(1.0)

    def test_weights_increase_toward_present(self):
        """Index 0 is oldest (smallest weight), index -1 newest (largest)."""
        w = _ewma_weights(halflife=10, window=50)
        assert np.all(np.diff(w) > 0)
        assert w[-1] > w[0]

    def test_halflife_property(self):
        """An observation `halflife` steps back gets half the newest weight."""
        hl, window = 8, 40
        w = _ewma_weights(halflife=hl, window=window)
        # newest is w[-1]; `hl` steps before newest is w[-1 - hl]
        assert w[-1 - hl] / w[-1] == pytest.approx(0.5)


class TestWeightedResidualize:
    """Tests for the `weights` argument to rolling_residualize (EWMA)."""

    def test_weighted_differs_from_unweighted(self):
        """Weighted residuals differ from equal-weight residuals."""
        np.random.seed(11)
        T = 80
        y = pd.DataFrame(np.random.randn(T, 2), columns=["y1", "y2"])
        X = pd.DataFrame(np.random.randn(T, 2), columns=["x1", "x2"])
        w = _ewma_weights(halflife=5, window=20)

        unweighted = rolling_residualize(y=y, X=X, window=20, min_periods=20, expanding=False)
        weighted = rolling_residualize(
            y=y, X=X, window=20, min_periods=20, expanding=False, weights=w
        )
        assert not np.allclose(unweighted.to_numpy(), weighted.to_numpy(), equal_nan=True)

    def test_weights_none_matches_baseline(self):
        """weights=None is bit-for-bit identical to omitting the argument."""
        np.random.seed(12)
        T = 60
        y = pd.DataFrame(np.random.randn(T, 3), columns=["y1", "y2", "y3"])
        X = pd.DataFrame(np.random.randn(T, 2), columns=["x1", "x2"])

        baseline = rolling_residualize(y=y, X=X, window=15, min_periods=15, expanding=False)
        explicit_none = rolling_residualize(
            y=y, X=X, window=15, min_periods=15, expanding=False, weights=None
        )
        pd.testing.assert_frame_equal(baseline, explicit_none)

    def test_uniform_weights_match_unweighted(self):
        """Uniform weights (all 1/window) reproduce the equal-weight result."""
        np.random.seed(13)
        T = 70
        y = pd.DataFrame(np.random.randn(T, 2), columns=["y1", "y2"])
        X = pd.DataFrame(np.random.randn(T, 2), columns=["x1", "x2"])
        window = 20
        uniform = np.full(window, 1.0 / window)

        unweighted = rolling_residualize(
            y=y, X=X, window=window, min_periods=window, expanding=False
        )
        weighted_uniform = rolling_residualize(
            y=y, X=X, window=window, min_periods=window, expanding=False, weights=uniform
        )
        np.testing.assert_allclose(
            unweighted.to_numpy(), weighted_uniform.to_numpy(), equal_nan=True, atol=1e-10
        )

    def test_weights_renormalized_after_nan_masking(self):
        """NaN-robust path (NaN in X): surviving weights renormalize to sum 1.

        With a single full window we can reproduce the residual by hand: drop
        the NaN row, renormalize the remaining weights to sum to 1, and solve
        the weighted least squares problem.
        """
        np.random.seed(14)
        window = 12
        T = window  # single window -> only the last timestep is defined
        y_np = np.random.randn(T)
        X_np = np.random.randn(T, 2)
        X_np[3, 0] = np.nan  # NaN in X invalidates row 3 -> per-column fallback

        y = pd.DataFrame(y_np, columns=["y"])
        X = pd.DataFrame(X_np, columns=["x1", "x2"])
        w = _ewma_weights(halflife=4, window=window)

        # min_periods <= surviving rows (11) so the window still produces a result.
        result = rolling_residualize(
            y=y, X=X, window=window, min_periods=window - 1, expanding=False, weights=w
        )

        # Manual weighted least squares on the surviving rows.
        row_ok = ~np.isnan(X_np).any(axis=1)
        w_c = w[row_ok]
        w_c = w_c / w_c.sum()
        assert w_c.sum() == pytest.approx(1.0)  # renormalized

        Xc = X_np[row_ok]
        yc = y_np[row_ok]
        XtX = Xc.T @ (Xc * w_c[:, None])
        beta = np.linalg.solve(XtX, Xc.T @ (yc * w_c))
        expected = y_np[-1] - X_np[-1] @ beta

        assert result.iloc[-1, 0] == pytest.approx(expected, rel=1e-10)

    def test_weighted_nan_in_y_matches_manual(self):
        """Intermediate path (NaN only in y): weighted residual matches manual WLS."""
        np.random.seed(15)
        window = 12
        T = window
        y_np = np.random.randn(T)
        X_np = np.random.randn(T, 2)
        y_np[5] = np.nan  # NaN in y -> intermediate vectorized path

        y = pd.DataFrame(y_np, columns=["y"])
        X = pd.DataFrame(X_np, columns=["x1", "x2"])
        w = _ewma_weights(halflife=4, window=window)

        result = rolling_residualize(
            y=y, X=X, window=window, min_periods=window - 1, expanding=False, weights=w
        )

        row_ok = ~np.isnan(y_np)
        w_c = w[row_ok]
        w_c = w_c / w_c.sum()
        Xc = X_np[row_ok]
        yc = y_np[row_ok]
        XtX = Xc.T @ (Xc * w_c[:, None])
        beta = np.linalg.solve(XtX, Xc.T @ (yc * w_c))
        expected = y_np[-1] - X_np[-1] @ beta

        assert result.iloc[-1, 0] == pytest.approx(expected, rel=1e-10)

    def test_weights_with_expanding_raises(self):
        """weights are not supported with expanding windows."""
        np.random.seed(16)
        T = 40
        y = pd.DataFrame(np.random.randn(T, 1), columns=["y"])
        X = pd.DataFrame(np.random.randn(T, 1), columns=["x"])
        w = _ewma_weights(halflife=5, window=20)
        with pytest.raises(ValueError, match="expanding"):
            rolling_residualize(y=y, X=X, window=20, min_periods=20, expanding=True, weights=w)


class TestHACSE:
    """Tests for HAC standard errors (Newey-West)."""

    def test_basic_hac_computation(self):
        """Test basic HAC standard error computation."""
        np.random.seed(42)
        T = 100
        residuals = pd.DataFrame(np.random.randn(T, 2), columns=["r1", "r2"])
        factor = pd.Series(np.random.randn(T), name="factor")

        result = hac_se(
            residuals=residuals,
            factor_values=factor,
            window=20,
            min_periods=20,
            expanding=False,
            n_lags=3,
        )

        assert result.shape == residuals.shape
        assert all(result.isna().sum() <= T // 2)  # Most values should be non-NaN

    def test_hac_expanding(self):
        """Test HAC with expanding window."""
        np.random.seed(42)
        T = 50
        residuals = pd.DataFrame(np.random.randn(T, 1), columns=["r"])
        factor = pd.Series(np.random.randn(T), name="factor")

        result = hac_se(
            residuals=residuals,
            factor_values=factor,
            window=10,
            min_periods=5,
            expanding=True,
            n_lags=2,
        )

        assert result.shape == residuals.shape

    def test_hac_with_nans(self):
        """Test HAC with NaN values."""
        np.random.seed(42)
        T = 100
        residuals = pd.DataFrame(np.random.randn(T, 1), columns=["r"])
        residuals.iloc[10:15] = np.nan
        factor = pd.Series(np.random.randn(T), name="factor")

        result = hac_se(
            residuals=residuals,
            factor_values=factor,
            window=20,
            min_periods=20,
            expanding=False,
            n_lags=3,
        )

        assert result.shape == residuals.shape

    def test_hac_non_negative_se(self):
        """Test that HAC SEs are non-negative."""
        np.random.seed(42)
        T = 100
        residuals = pd.DataFrame(np.random.randn(T, 2), columns=["r1", "r2"])
        factor = pd.Series(np.random.randn(T), name="factor")

        result = hac_se(
            residuals=residuals,
            factor_values=factor,
            window=20,
            min_periods=20,
            expanding=False,
            n_lags=3,
        )

        # All non-NaN values should be non-negative
        assert (result.fillna(0) >= 0).all().all()

    def test_different_lag_lengths(self):
        """Test HAC with different lag specifications."""
        np.random.seed(42)
        T = 100
        residuals = pd.DataFrame(np.random.randn(T, 1), columns=["r"])
        factor = pd.Series(np.random.randn(T), name="factor")

        result_1lag = hac_se(
            residuals=residuals,
            factor_values=factor,
            window=20,
            min_periods=20,
            expanding=False,
            n_lags=1,
        )

        result_5lag = hac_se(
            residuals=residuals,
            factor_values=factor,
            window=20,
            min_periods=20,
            expanding=False,
            n_lags=5,
        )

        # Different lag lengths should produce different results
        assert not result_1lag.equals(result_5lag)

    def test_nan_isolated_per_asset(self):
        """A NaN in one asset must not contaminate other assets' SE (issue #8)."""
        np.random.seed(42)
        T = 100
        clean = pd.DataFrame(np.random.randn(T, 3), columns=["a1", "a2", "a3"])
        factor = pd.Series(np.random.randn(T), name="factor")

        se_clean = hac_se(
            residuals=clean,
            factor_values=factor,
            window=20,
            min_periods=20,
            expanding=False,
            n_lags=3,
        )

        # Introduce a NaN into a2 only, inside one window's reach.
        contaminated = clean.copy()
        contaminated.iloc[50, contaminated.columns.get_loc("a2")] = np.nan

        se_contam = hac_se(
            residuals=contaminated,
            factor_values=factor,
            window=20,
            min_periods=20,
            expanding=False,
            n_lags=3,
        )

        # a1 and a3 SEs are identical to the all-clean run — no contamination.
        pd.testing.assert_series_equal(se_clean["a1"], se_contam["a1"])
        pd.testing.assert_series_equal(se_clean["a3"], se_contam["a3"])

        # a2 is NaN exactly on the windows that span the NaN row (t in [50, 69]),
        # and unaffected outside that span.
        affected = se_contam["a2"].iloc[50:70]
        assert affected.isna().all()
        # A window ending before the NaN row is still valid for a2.
        assert not np.isnan(se_contam["a2"].iloc[49])

    def test_factor_nan_invalidates_whole_window(self):
        """A factor NaN drops SE for ALL assets in affected windows."""
        np.random.seed(0)
        T = 100
        residuals = pd.DataFrame(np.random.randn(T, 3), columns=["a1", "a2", "a3"])
        factor = pd.Series(np.random.randn(T), name="factor")
        factor.iloc[50] = np.nan

        se = hac_se(
            residuals=residuals,
            factor_values=factor,
            window=20,
            min_periods=20,
            expanding=False,
            n_lags=3,
        )

        # Every window spanning t=50 (i.e. t in [50, 69]) is fully NaN.
        affected = se.iloc[50:70]
        assert affected.isna().all().all()
        # A window ending at t=49 is unaffected for all assets.
        assert se.iloc[49].notna().all()

    def test_matches_manual_newey_west(self):
        """Cross-check SE against a manual Newey-West on a clean single-asset window."""
        np.random.seed(7)
        T = 40
        window, n_lags = 20, 3
        e = np.random.randn(T)
        f = np.random.randn(T)
        residuals = pd.DataFrame({"a1": e})
        factor = pd.Series(f, name="factor")

        se = hac_se(
            residuals=residuals,
            factor_values=factor,
            window=window,
            min_periods=window,
            expanding=False,
            n_lags=n_lags,
        )

        # Manual Newey-West for the window ending at the last timestep.
        t = T - 1
        f_w = f[t - window + 1 : t + 1]
        e_w = e[t - window + 1 : t + 1]
        n_obs = window
        score = f_w * e_w
        xx = f_w @ f_w
        S = np.sum(score**2) / n_obs
        for lag in range(1, n_lags + 1):
            w = 1.0 - lag / (n_lags + 1)
            gamma = np.sum(score[lag:] * score[:-lag]) / n_obs
            S += 2 * w * gamma
        var_beta = S * n_obs / (xx**2)
        expected = np.sqrt(max(var_beta, 0.0))

        assert se["a1"].iloc[t] == pytest.approx(expected, rel=1e-10)

    @staticmethod
    def _loop_hac_se(residuals, factor_values, window, min_periods, n_lags):
        """Reference per-window loop implementation (the pre-vectorization path)."""
        resid_np = residuals.to_numpy(dtype=np.float64)
        f_np = factor_values.to_numpy(dtype=np.float64)
        T, N = resid_np.shape
        se = np.full((T, N), np.nan)

        def _nw(f_w, e_w):
            n_obs = len(f_w)
            score = f_w[:, None] * e_w
            xx = f_w @ f_w
            S = np.einsum("ti,ti->i", score, score) / n_obs
            for lag in range(1, n_lags + 1):
                w = 1.0 - lag / (n_lags + 1)
                gamma = np.einsum("ti,ti->i", score[lag:], score[:-lag]) / n_obs
                S += 2 * w * gamma
            return np.sqrt(np.maximum(S * n_obs / (xx**2), 0.0))

        def _fill(t, f_w, e_w):
            if np.isnan(f_w).any() or len(f_w) <= n_lags:
                return
            asset_nan = np.isnan(e_w).any(axis=0)
            if asset_nan.all():
                return
            valid = ~asset_nan
            se[t, valid] = _nw(f_w, e_w[:, valid])

        for t in range(window - 1, T):
            start = t - window + 1
            _fill(t, f_np[start : t + 1], resid_np[start : t + 1])
        if min_periods < window:
            for t in range(min_periods - 1, window - 1):
                _fill(t, f_np[: t + 1], resid_np[: t + 1])
        return pd.DataFrame(se, index=residuals.index, columns=residuals.columns)

    def test_vectorized_matches_loop(self):
        """Vectorized rolling path is numerically identical to the loop path."""
        np.random.seed(123)
        T, N = 200, 5
        residuals = pd.DataFrame(np.random.randn(T, N), columns=[f"a{i}" for i in range(N)])
        factor = pd.Series(np.random.randn(T), name="factor")

        for window, n_lags in [(20, 3), (30, 5), (15, 1), (50, 8)]:
            got = hac_se(
                residuals=residuals,
                factor_values=factor,
                window=window,
                min_periods=window,
                expanding=False,
                n_lags=n_lags,
            )
            ref = self._loop_hac_se(residuals, factor, window, window, n_lags)
            assert np.allclose(got.to_numpy(), ref.to_numpy(), equal_nan=True), (
                f"mismatch for window={window}, n_lags={n_lags}"
            )

    def test_vectorized_matches_loop_with_nans(self):
        """Vectorized path matches the loop when factor and residual NaNs are present."""
        np.random.seed(321)
        T, N = 150, 4
        residuals = pd.DataFrame(np.random.randn(T, N), columns=[f"a{i}" for i in range(N)])
        residuals.iloc[40, 1] = np.nan  # per-asset residual NaN
        residuals.iloc[80:83, 2] = np.nan
        factor = pd.Series(np.random.randn(T), name="factor")
        factor.iloc[100] = np.nan  # whole-window factor NaN

        got = hac_se(
            residuals=residuals,
            factor_values=factor,
            window=25,
            min_periods=25,
            expanding=False,
            n_lags=4,
        )
        ref = self._loop_hac_se(residuals, factor, 25, 25, 4)
        assert np.allclose(got.to_numpy(), ref.to_numpy(), equal_nan=True)

    def test_vectorized_matches_loop_min_periods_lt_window(self):
        """Early (min_periods < window) windows still match the loop path."""
        np.random.seed(7)
        T, N = 120, 3
        residuals = pd.DataFrame(np.random.randn(T, N), columns=[f"a{i}" for i in range(N)])
        factor = pd.Series(np.random.randn(T), name="factor")

        got = hac_se(
            residuals=residuals,
            factor_values=factor,
            window=30,
            min_periods=15,
            expanding=False,
            n_lags=3,
        )
        ref = self._loop_hac_se(residuals, factor, 30, 15, 3)
        assert np.allclose(got.to_numpy(), ref.to_numpy(), equal_nan=True)

    def test_expanding_matches_loop(self):
        """Expanding path is unchanged (still the loop) and matches the reference."""
        np.random.seed(11)
        T, N = 80, 2
        residuals = pd.DataFrame(np.random.randn(T, N), columns=["a0", "a1"])
        factor = pd.Series(np.random.randn(T), name="factor")

        # Reference expanding loop.
        resid_np = residuals.to_numpy(dtype=np.float64)
        f_np = factor.to_numpy(dtype=np.float64)
        n_lags, min_periods = 3, 10
        ref = np.full((T, N), np.nan)
        for t in range(min_periods - 1, T):
            f_w, e_w = f_np[: t + 1], resid_np[: t + 1]
            if np.isnan(f_w).any() or len(f_w) <= n_lags:
                continue
            score = f_w[:, None] * e_w
            xx = f_w @ f_w
            S = np.einsum("ti,ti->i", score, score) / len(f_w)
            for lag in range(1, n_lags + 1):
                w = 1.0 - lag / (n_lags + 1)
                S += 2 * w * np.einsum("ti,ti->i", score[lag:], score[:-lag]) / len(f_w)
            ref[t] = np.sqrt(np.maximum(S * len(f_w) / (xx**2), 0.0))

        got = hac_se(
            residuals=residuals,
            factor_values=factor,
            window=40,
            min_periods=min_periods,
            expanding=True,
            n_lags=n_lags,
        )
        assert np.allclose(got.to_numpy(), ref, equal_nan=True)


class TestSolveBatch:
    """Tests for _solve_batch inf/NaN handling (issue #6)."""

    def test_well_conditioned_correct_betas(self):
        """Well-conditioned windows return the exact OLS solution."""
        np.random.seed(0)
        n, k, N = 4, 2, 3
        XtX = np.empty((n, k, k))
        XtY = np.empty((n, k, N))
        expected = np.empty((n, k, N))
        for i in range(n):
            A = np.random.randn(k, k)
            XtX_i = A @ A.T + np.eye(k)  # SPD, well-conditioned
            beta_i = np.random.randn(k, N)
            XtX[i] = XtX_i
            XtY[i] = XtX_i @ beta_i
            expected[i] = beta_i

        betas = _solve_batch(XtX, XtY)
        assert np.isfinite(betas).all()
        np.testing.assert_allclose(betas, expected, rtol=1e-8)

    def test_non_finite_solve_returns_nan(self):
        """A window whose solve overflows to inf is sanitized to NaN, not inf."""
        # diag(1e-160) with huge RHS overflows in np.linalg.solve without
        # raising LinAlgError -> result contains inf/nan.
        XtX = np.array([[[1e-160, 0.0], [0.0, 1e-160]]])
        XtY = np.array([[[1e200], [1e200]]])

        betas = _solve_batch(XtX, XtY)
        assert not np.isinf(betas).any()
        assert np.isnan(betas).all()

    def test_singular_window_returns_nan(self):
        """An exactly singular window returns NaN (not inf), via the fallback."""
        # Collinear columns -> singular XtX
        XtX = np.array([[[1.0, 1.0], [1.0, 1.0]]])
        XtY = np.array([[[1.0], [2.0]]])

        betas = _solve_batch(XtX, XtY, warn_singular=False)
        assert not np.isinf(betas).any()
        assert np.isnan(betas).all()

    def test_mixed_batch_isolates_bad_window(self):
        """A bad window stays NaN while good windows in the same batch solve correctly."""
        good_XtX = np.array([[2.0, 0.0], [0.0, 4.0]])
        good_beta = np.array([[1.5], [-2.0]])
        good_XtY = good_XtX @ good_beta

        # singular second window forces the element-wise fallback for the batch
        bad_XtX = np.array([[1.0, 1.0], [1.0, 1.0]])
        bad_XtY = np.array([[1.0], [2.0]])

        XtX = np.stack([good_XtX, bad_XtX])
        XtY = np.stack([good_XtY, bad_XtY])

        betas = _solve_batch(XtX, XtY, warn_singular=False)
        np.testing.assert_allclose(betas[0], good_beta, rtol=1e-10)
        assert np.isnan(betas[1]).all()


class TestSingularWarnings:
    """Tests for singular-matrix warnings (issue #12)."""

    def _singular_inputs(self, T=60):
        """X with duplicate columns -> singular windows; clean (no NaN) fast path."""
        np.random.seed(42)
        x = np.random.randn(T)
        # two identical columns -> X'X singular
        X = pd.DataFrame({"x1": x, "x2": x})
        y = pd.DataFrame(np.random.randn(T, 1), columns=["y1"])
        return y, X

    def test_singular_emits_warning(self):
        """Singular windows trigger a RuntimeWarning (not silent)."""
        y, X = self._singular_inputs()
        with pytest.warns(RuntimeWarning, match="singular"):
            rolling_residualize(
                y=y, X=X, window=20, min_periods=20, expanding=False, ridge_lambda=0.0
            )

    def test_singular_output_is_nan_not_inf(self):
        """Singular windows produce NaN, not inf, and don't raise."""
        y, X = self._singular_inputs()
        with pytest.warns(RuntimeWarning):
            result = rolling_residualize(
                y=y, X=X, window=20, min_periods=20, expanding=False, ridge_lambda=0.0
            )
        vals = result.to_numpy()
        assert not np.isinf(vals).any()
        # The windowed region is singular -> NaN there.
        assert np.isnan(vals[19:]).all()

    def test_warn_singular_false_suppresses(self, recwarn):
        """warn_singular=False suppresses the warning."""
        y, X = self._singular_inputs()
        rolling_residualize(
            y=y,
            X=X,
            window=20,
            min_periods=20,
            expanding=False,
            ridge_lambda=0.0,
            warn_singular=False,
        )
        assert len(recwarn) == 0

    def test_non_singular_no_warning(self, recwarn):
        """Well-conditioned input emits no warning."""
        np.random.seed(0)
        T = 60
        y = pd.DataFrame(np.random.randn(T, 2), columns=["y1", "y2"])
        X = pd.DataFrame(np.random.randn(T, 2), columns=["x1", "x2"])
        rolling_residualize(y=y, X=X, window=20, min_periods=20, expanding=False, ridge_lambda=0.0)
        assert not any(issubclass(w.category, RuntimeWarning) for w in recwarn)

    def test_singular_warning_expanding_path(self):
        """The expanding-window path also warns on singular windows."""
        y, X = self._singular_inputs()
        with pytest.warns(RuntimeWarning, match="singular"):
            rolling_residualize(
                y=y, X=X, window=20, min_periods=20, expanding=True, ridge_lambda=0.0
            )
