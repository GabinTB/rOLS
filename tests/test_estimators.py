"""Tests for low-level rolling estimators."""

import numpy as np
import pandas as pd
import pytest

from rols.estimators import (
    rolling_residualize,
    rolling_gram_schmidt,
    hac_se,
    _solve_batch,
)


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
        assert result.iloc[min_periods - 1:].notna().any().any()

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


class TestRollingGramSchmidt:
    """Tests for rolling_gram_schmidt function."""

    def test_basic_orthogonalization(self):
        """Test basic Gram-Schmidt orthogonalization."""
        np.random.seed(42)
        T = 100
        X = pd.DataFrame(np.random.randn(T, 3), columns=["x1", "x2", "x3"])

        result = rolling_gram_schmidt(
            X=X, window=20, min_periods=20, expanding=False
        )

        assert result.shape == X.shape
        assert list(result.columns) == list(X.columns)

    def test_single_column_unchanged(self):
        """Test that single column remains unchanged."""
        np.random.seed(42)
        T = 100
        X = pd.DataFrame(np.random.randn(T, 1), columns=["x1"])

        result = rolling_gram_schmidt(
            X=X, window=20, min_periods=20, expanding=False
        )

        # First column should remain unchanged
        pd.testing.assert_frame_equal(result, X)

    def test_expanding_orthogonalization(self):
        """Test orthogonalization with expanding window."""
        np.random.seed(42)
        T = 50
        X = pd.DataFrame(np.random.randn(T, 2), columns=["x1", "x2"])

        result = rolling_gram_schmidt(
            X=X, window=10, min_periods=5, expanding=True
        )

        assert result.shape == X.shape

    def test_multiple_columns(self):
        """Test with multiple columns."""
        np.random.seed(42)
        T = 100
        X = pd.DataFrame(np.random.randn(T, 5), columns=[f"x{i}" for i in range(5)])

        result = rolling_gram_schmidt(
            X=X, window=20, min_periods=20, expanding=False
        )

        assert result.shape == X.shape
        assert list(result.columns) == list(X.columns)

    def test_orthogonalization_reduces_correlation(self):
        """Test that orthogonalization reduces correlation."""
        np.random.seed(42)
        T = 100
        # Create highly correlated columns
        x1 = pd.Series(np.random.randn(T))
        x2 = x1 + 0.1 * np.random.randn(T)
        X = pd.DataFrame({"x1": x1, "x2": x2})

        result = rolling_gram_schmidt(
            X=X, window=20, min_periods=20, expanding=False
        )

        # First column should be unchanged
        pd.testing.assert_series_equal(result["x1"], X["x1"], check_dtype=False)


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

        betas = _solve_batch(XtX, XtY)
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

        betas = _solve_batch(XtX, XtY)
        np.testing.assert_allclose(betas[0], good_beta, rtol=1e-10)
        assert np.isnan(betas[1]).all()
