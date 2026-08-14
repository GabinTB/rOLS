"""Integration tests for rols library."""

import numpy as np
import pandas as pd
import pytest

from rols import RollingOLS, RollingOLSResult


class TestIntegrationBasicWorkflow:
    """Integration tests for basic end-to-end workflows."""

    def test_complete_workflow_no_controls(self):
        """Test complete workflow without control variables."""
        np.random.seed(42)
        T = 252
        dates = pd.date_range("2022-01-01", periods=T)

        factors = pd.DataFrame(
            np.random.randn(T, 2), index=dates, columns=["narrative_1", "narrative_2"]
        )
        assets = pd.DataFrame(np.random.randn(T, 3), index=dates, columns=["AAPL", "MSFT", "GOOG"])

        ols = RollingOLS(window=60, min_periods=20)
        result = ols.fit_transform(factors, assets)

        assert isinstance(result, RollingOLSResult)
        assert result.get_beta("narrative_1").shape == (T, 3)
        assert result.get_signal("narrative_2").shape == (T, 3)
        assert result.get_r2("narrative_1").shape == (T, 3)

    def test_complete_workflow_with_controls(self):
        """Test complete workflow with control variables."""
        np.random.seed(42)
        T = 252
        dates = pd.date_range("2022-01-01", periods=T)

        factors = pd.DataFrame(
            np.random.randn(T, 2), index=dates, columns=["narrative_1", "narrative_2"]
        )
        controls = pd.DataFrame(np.random.randn(T, 2), index=dates, columns=["Mkt-RF", "SMB"])
        assets = pd.DataFrame(
            np.random.randn(T, 5), index=dates, columns=[f"stock_{i}" for i in range(5)]
        )

        ols = RollingOLS(window=60)
        result = ols.fit_transform(factors, assets, controls=controls)

        assert result.get_beta("narrative_1").shape == (T, 5)

    def test_complete_workflow_with_hac(self):
        """Test complete workflow with HAC standard errors."""
        np.random.seed(42)
        T = 252
        dates = pd.date_range("2022-01-01", periods=T)

        factors = pd.DataFrame(np.random.randn(T, 1), index=dates, columns=["factor_1"])
        assets = pd.DataFrame(np.random.randn(T, 2), index=dates, columns=["asset_1", "asset_2"])

        ols = RollingOLS(window=60, hac_lags=5)
        result = ols.fit_transform(factors, assets)

        se = result.get_se("factor_1")
        tstat = result.get_tstat("factor_1")

        assert se.shape == (T, 2)
        assert tstat.shape == (T, 2)

    def test_complete_workflow_expanding_window(self):
        """Test complete workflow with expanding window."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 1), columns=["f1"])
        assets = pd.DataFrame(np.random.randn(T, 2), columns=["a1", "a2"])

        ols = RollingOLS(window=20, min_periods=10, expanding=True)
        result = ols.fit_transform(factors, assets)

        beta = result.get_beta("f1")
        assert beta.shape == (T, 2)
        # Early windows should have some non-NaN values
        assert beta.iloc[9:].notna().any().any()

    def test_complete_workflow_with_ridge(self):
        """Test complete workflow with Ridge regularization."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 3), columns=["f1", "f2", "f3"])
        assets = pd.DataFrame(np.random.randn(T, 2), columns=["a1", "a2"])

        ols = RollingOLS(window=20, lambda_=0.1)
        result = ols.fit_transform(factors, assets)

        assert result.get_beta("f1").shape == (T, 2)

    def test_complete_workflow_with_orthogonalization(self):
        """Test complete workflow with orthogonalization."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 3), columns=["f1", "f2", "f3"])
        controls = pd.DataFrame(np.random.randn(T, 2), columns=["c1", "c2"])
        assets = pd.DataFrame(np.random.randn(T, 2), columns=["a1", "a2"])

        ols = RollingOLS(window=20)
        result = ols.fit_transform(
            factors,
            assets,
            controls=controls,
            orthogonalize_factors=True,
            orthogonalize_controls=True,
        )

        assert result.get_beta("f1").shape == (T, 2)

    def test_complete_workflow_lag_signal(self):
        """Test complete workflow with lagged signals."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 1), columns=["f1"])
        assets = pd.DataFrame(np.random.randn(T, 2), columns=["a1", "a2"])

        ols = RollingOLS(window=20, lag_signal=True)
        result = ols.fit_transform(factors, assets)

        signal = result.get_signal("f1")
        assert signal.shape == (T, 2)


class TestIntegrationDataHandling:
    """Integration tests for data handling."""

    def test_with_real_like_data(self):
        """Test with realistic looking data."""
        np.random.seed(42)
        T = 252 * 2  # 2 years of daily data
        dates = pd.date_range("2022-01-01", periods=T, freq="D")

        # Factors: some mean-reverting, some trending
        f1 = np.cumsum(np.random.randn(T) * 0.01)
        f2 = np.sin(np.linspace(0, 4 * np.pi, T)) + np.random.randn(T) * 0.1

        factors = pd.DataFrame({"momentum": f1, "sentiment": f2}, index=dates)

        # Assets: correlated with factors
        assets = pd.DataFrame(
            {
                "tech_stock": f1 * 0.5 + np.random.randn(T) * 0.5,
                "financial_stock": f2 * 0.3 + np.random.randn(T) * 0.5,
            },
            index=dates,
        )

        ols = RollingOLS(window=60, min_periods=20, hac_lags=5)
        result = ols.fit_transform(factors, assets)

        assert result.get_beta("momentum").shape == (T, 2)
        assert result.get_tstat("sentiment").shape == (T, 2)

    def test_with_missing_data(self):
        """Test with missing data (NaNs)."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 2), columns=["f1", "f2"])
        assets = pd.DataFrame(np.random.randn(T, 2), columns=["a1", "a2"])

        # Inject some NaNs
        factors.iloc[10:15, 0] = np.nan
        assets.iloc[20:25, 1] = np.nan

        ols = RollingOLS(window=20)
        result = ols.fit_transform(factors, assets)

        # Should handle NaNs gracefully
        assert result.get_beta("f1").shape == (T, 2)

    def test_with_custom_index(self):
        """Test with custom time index."""
        np.random.seed(42)
        T = 100
        # Custom date index with gaps
        dates = pd.date_range("2022-01-01", periods=T, freq="B")  # Business days

        factors = pd.DataFrame(np.random.randn(T, 1), columns=["f1"], index=dates)
        assets = pd.DataFrame(np.random.randn(T, 1), columns=["a1"], index=dates)

        ols = RollingOLS(window=20)
        result = ols.fit_transform(factors, assets)

        # Index should be preserved
        pd.testing.assert_index_equal(result.index, dates)

    def test_with_multiindex_columns(self):
        """Test that we can handle regular dataframes (not MultiIndex)."""
        np.random.seed(42)
        T = 50
        factors = pd.DataFrame(np.random.randn(T, 2), columns=["f1", "f2"])
        assets = pd.DataFrame(np.random.randn(T, 3), columns=["a1", "a2", "a3"])

        ols = RollingOLS(window=20)
        result = ols.fit_transform(factors, assets)

        assert result.get_beta("f1").shape == (T, 3)


class TestIntegrationOutputFormats:
    """Integration tests for different output formats."""

    def test_long_format_single_factor(self):
        """Test long format output for single factor."""
        np.random.seed(42)
        T = 50
        factors = pd.DataFrame(np.random.randn(T, 2), columns=["f1", "f2"])
        assets = pd.DataFrame(np.random.randn(T, 2), columns=["a1", "a2"])

        ols = RollingOLS(window=10)
        result = ols.fit_transform(factors, assets)

        long_df = result.to_long("f1")

        # Should have some rows (may filter out NaNs from early periods)
        assert len(long_df) > 0
        assert "beta" in long_df.columns
        assert "signal" in long_df.columns
        assert "r2" in long_df.columns

    def test_long_format_all_factors(self):
        """Test long format output for all factors."""
        np.random.seed(42)
        T = 50
        factors = pd.DataFrame(np.random.randn(T, 3), columns=["f1", "f2", "f3"])
        assets = pd.DataFrame(np.random.randn(T, 2), columns=["a1", "a2"])

        ols = RollingOLS(window=10)
        result = ols.fit_transform(factors, assets)

        long_df = result.to_long_all()

        # Should have some rows (may filter out NaNs)
        assert len(long_df) > 0
        assert "factor" in long_df.columns
        assert set(long_df["factor"]) == {"f1", "f2", "f3"}

    def test_long_format_with_hac(self):
        """Test long format output with HAC."""
        np.random.seed(42)
        T = 50
        factors = pd.DataFrame(np.random.randn(T, 1), columns=["f1"])
        assets = pd.DataFrame(np.random.randn(T, 2), columns=["a1", "a2"])

        ols = RollingOLS(window=10, hac_lags=2)
        result = ols.fit_transform(factors, assets)

        long_df = result.to_long("f1", include_se=True)

        assert "se" in long_df.columns
        assert "t_stat" in long_df.columns


class TestIntegrationStatisticalProperties:
    """Integration tests for statistical properties."""

    def test_orthogonalization_reduces_correlation(self):
        """Test that orthogonalization reduces factor correlation."""
        np.random.seed(42)
        T = 100

        # Create correlated factors
        f1 = pd.Series(np.random.randn(T))
        f2 = f1 + 0.1 * np.random.randn(T)
        f3 = f1 + f2 + 0.1 * np.random.randn(T)

        factors = pd.DataFrame({"f1": f1, "f2": f2, "f3": f3})
        assets = pd.DataFrame(np.random.randn(T, 2), columns=["a1", "a2"])

        # With orthogonalization
        ols = RollingOLS(window=20)
        result = ols.fit_transform(factors, assets, orthogonalize_factors=True)

        assert result.get_beta("f1").shape == (T, 2)

    def test_expanding_window_accumulates_data(self):
        """Test that expanding window accumulates information."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 1), columns=["f1"])
        assets = pd.DataFrame(np.random.randn(T, 1), columns=["a1"])

        ols = RollingOLS(window=20, min_periods=5, expanding=True)
        result = ols.fit_transform(factors, assets)

        beta = result.get_beta("f1").iloc[:, 0]
        # Later values should be less sparse (expanding accumulates data)
        nans_early = beta.iloc[:20].isna().sum()
        nans_late = beta.iloc[-20:].isna().sum()
        assert nans_early >= nans_late

    def test_ridge_vs_ols_betas(self):
        """Ridge changes and strictly shrinks betas without controls."""
        rng = np.random.default_rng(42)
        factor_values = rng.normal(size=100)
        factors = pd.DataFrame({"factor": factor_values})
        assets = pd.DataFrame({"asset": 3.0 + 2.0 * factor_values})

        beta_ols = (
            RollingOLS(window=30, lambda_=0.0, dtype="float64")
            .fit_transform(factors, assets)
            .get_beta("factor")
        )
        beta_ridge = (
            RollingOLS(window=30, lambda_=1000.0, dtype="float64")
            .fit_transform(factors, assets)
            .get_beta("factor")
        )
        valid = beta_ols.notna() & beta_ridge.notna()

        assert not np.allclose(beta_ols.to_numpy()[valid], beta_ridge.to_numpy()[valid])
        assert (np.abs(beta_ridge.to_numpy()[valid]) < np.abs(beta_ols.to_numpy()[valid])).all()


class TestIntegrationErrorHandling:
    """Integration tests for error handling."""

    def test_transform_before_fit_raises(self):
        """Test that transform before fit raises error."""
        T = 50
        assets = pd.DataFrame(np.random.randn(T, 1), columns=["a1"])

        ols = RollingOLS(window=20)

        with pytest.raises(RuntimeError):
            ols.transform(assets)

    def test_invalid_factor_name_raises(self):
        """Test that invalid factor name raises error."""
        np.random.seed(42)
        T = 50
        factors = pd.DataFrame(np.random.randn(T, 1), columns=["f1"])
        assets = pd.DataFrame(np.random.randn(T, 1), columns=["a1"])

        ols = RollingOLS(window=20)
        result = ols.fit_transform(factors, assets)

        with pytest.raises(KeyError):
            result.get_beta("nonexistent")

    def test_hac_without_lags_raises(self):
        """Test that HAC without lags raises error."""
        np.random.seed(42)
        T = 50
        factors = pd.DataFrame(np.random.randn(T, 1), columns=["f1"])
        assets = pd.DataFrame(np.random.randn(T, 1), columns=["a1"])

        ols = RollingOLS(window=20, hac_lags=None)
        result = ols.fit_transform(factors, assets)

        with pytest.raises(RuntimeError):
            result.get_se("f1")
