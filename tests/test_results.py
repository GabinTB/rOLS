"""Tests for the RollingOLSResult class."""

import numpy as np
import pandas as pd
import pytest

from rols.model import RollingOLS
from rols.results import RollingOLSResult


class TestRollingOLSResultGetters:
    """Tests for result getter methods."""

    @pytest.fixture
    def setup_result(self):
        """Create a fitted RollingOLS result for testing."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 2), columns=["f1", "f2"])
        assets = pd.DataFrame(np.random.randn(T, 3), columns=["a1", "a2", "a3"])

        ols = RollingOLS(window=20, hac_lags=3)
        result = ols.fit_transform(factors, assets)
        return result

    def test_get_beta(self, setup_result):
        """Test get_beta method."""
        result = setup_result
        beta = result.get_beta("f1")

        assert isinstance(beta, pd.DataFrame)
        assert beta.shape == (100, 3)
        assert list(beta.columns) == ["a1", "a2", "a3"]

    def test_get_signal(self, setup_result):
        """Test get_signal method."""
        result = setup_result
        signal = result.get_signal("f1")

        assert isinstance(signal, pd.DataFrame)
        assert signal.shape == (100, 3)

    def test_get_r2(self, setup_result):
        """Test get_r2 method."""
        result = setup_result
        r2 = result.get_r2("f1")

        assert isinstance(r2, pd.DataFrame)
        assert r2.shape == (100, 3)
        # R² should be between 0 and 1 (where not NaN)
        r2_valid = r2.dropna().values.flatten()
        if len(r2_valid) > 0:
            assert (r2_valid >= 0).all() and (r2_valid <= 1).all()

    def test_get_residuals(self, setup_result):
        """Test get_residuals method."""
        result = setup_result
        resids = result.get_residuals("f1")

        assert isinstance(resids, pd.DataFrame)
        assert resids.shape == (100, 3)

    def test_get_invalid_factor_raises(self, setup_result):
        """Test that invalid factor name raises KeyError."""
        result = setup_result

        with pytest.raises(KeyError):
            result.get_beta("invalid_factor")

    def test_factor_cols_property(self, setup_result):
        """Test factor_cols property."""
        result = setup_result
        assert list(result.factor_cols) == ["f1", "f2"]

    def test_asset_cols_property(self, setup_result):
        """Test asset_cols property."""
        result = setup_result
        assert list(result.asset_cols) == ["a1", "a2", "a3"]


class TestRollingOLSResultHAC:
    """Tests for HAC standard errors and t-statistics."""

    @pytest.fixture
    def setup_result_with_hac(self):
        """Create result with HAC enabled."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 1), columns=["f1"])
        assets = pd.DataFrame(np.random.randn(T, 2), columns=["a1", "a2"])

        ols = RollingOLS(window=20, hac_lags=5)
        result = ols.fit_transform(factors, assets)
        return result

    @pytest.fixture
    def setup_result_no_hac(self):
        """Create result without HAC enabled."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 1), columns=["f1"])
        assets = pd.DataFrame(np.random.randn(T, 2), columns=["a1", "a2"])

        ols = RollingOLS(window=20, hac_lags=None)
        result = ols.fit_transform(factors, assets)
        return result

    def test_get_se_basic(self, setup_result_with_hac):
        """Test get_se method."""
        result = setup_result_with_hac
        se = result.get_se("f1")

        assert isinstance(se, pd.DataFrame)
        assert se.shape == (100, 2)
        # SEs should be non-negative
        se_valid = se.dropna().values.flatten()
        if len(se_valid) > 0:
            assert (se_valid >= 0).all()

    def test_get_se_without_hac_raises(self, setup_result_no_hac):
        """Test that get_se raises without hac_lags."""
        result = setup_result_no_hac

        with pytest.raises(RuntimeError):
            result.get_se("f1")

    def test_get_tstat_basic(self, setup_result_with_hac):
        """Test get_tstat method."""
        result = setup_result_with_hac
        tstat = result.get_tstat("f1")

        assert isinstance(tstat, pd.DataFrame)
        assert tstat.shape == (100, 2)

    def test_get_tstat_without_hac_raises(self, setup_result_no_hac):
        """Test that get_tstat raises without hac_lags."""
        result = setup_result_no_hac

        with pytest.raises(RuntimeError):
            result.get_tstat("f1")

    def test_se_cached(self, setup_result_with_hac):
        """Test that SE is cached after first call."""
        result = setup_result_with_hac

        # First call
        se1 = result.get_se("f1")
        # Second call should return same object from cache
        se2 = result.get_se("f1")

        assert se1 is se2  # Same object reference

    def test_tstat_uses_cached_se(self, setup_result_with_hac):
        """Test that get_tstat uses cached SE."""
        result = setup_result_with_hac

        # Get SE first
        se = result.get_se("f1")
        # Get tstat
        tstat = result.get_tstat("f1")
        beta = result.get_beta("f1")

        # tstat should equal beta / se
        expected_tstat = beta / se
        pd.testing.assert_frame_equal(tstat, expected_tstat, check_dtype=False)


class TestRollingOLSResultLongFormat:
    """Tests for long-format output methods."""

    @pytest.fixture
    def setup_result(self):
        """Create a fitted RollingOLS result."""
        np.random.seed(42)
        T = 50
        factors = pd.DataFrame(np.random.randn(T, 2), columns=["f1", "f2"])
        assets = pd.DataFrame(np.random.randn(T, 2), columns=["a1", "a2"])

        ols = RollingOLS(window=10, hac_lags=2)
        result = ols.fit_transform(factors, assets)
        return result

    def test_to_long_basic(self, setup_result):
        """Test to_long method."""
        result = setup_result
        long_df = result.to_long("f1")

        assert isinstance(long_df, pd.DataFrame)
        assert "date" in long_df.columns
        assert "asset" in long_df.columns
        assert "beta" in long_df.columns
        assert "signal" in long_df.columns
        assert "r2" in long_df.columns
        # Should have at least some rows (may filter out NaNs)
        assert len(long_df) > 0

    def test_to_long_with_se(self, setup_result):
        """Test to_long with include_se=True."""
        result = setup_result
        long_df = result.to_long("f1", include_se=True)

        assert "se" in long_df.columns
        assert "t_stat" in long_df.columns

    def test_to_long_invalid_factor_raises(self, setup_result):
        """Test that invalid factor raises KeyError."""
        result = setup_result

        with pytest.raises(KeyError):
            result.to_long("invalid")

    def test_to_long_all(self, setup_result):
        """Test to_long_all method."""
        result = setup_result
        long_df = result.to_long_all()

        assert isinstance(long_df, pd.DataFrame)
        assert "factor" in long_df.columns
        # Should have results for all factors
        assert set(long_df["factor"]) == {"f1", "f2"}
        # Total rows should be at least some percentage of T * N_assets * N_factors
        assert len(long_df) > 0

    def test_to_long_all_with_se(self, setup_result):
        """Test to_long_all with include_se=True."""
        result = setup_result
        long_df = result.to_long_all(include_se=True)

        assert "se" in long_df.columns
        assert "t_stat" in long_df.columns

    def test_to_long_column_order(self, setup_result):
        """Test that to_long has expected column order."""
        result = setup_result
        long_df = result.to_long("f1", include_se=False)

        expected_cols = ["date", "asset", "beta", "signal", "r2"]
        assert list(long_df.columns) == expected_cols

    def test_to_long_all_column_order(self, setup_result):
        """Test that to_long_all has expected column order."""
        result = setup_result
        long_df = result.to_long_all(include_se=False)

        expected_cols = ["date", "asset", "factor", "beta", "signal", "r2"]
        assert list(long_df.columns) == expected_cols


class TestRollingOLSResultConsistency:
    """Tests for consistency between different result methods."""

    @pytest.fixture
    def setup_result(self):
        """Create a result for consistency tests."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 2), columns=["f1", "f2"])
        assets = pd.DataFrame(np.random.randn(T, 2), columns=["a1", "a2"])

        ols = RollingOLS(window=20)
        result = ols.fit_transform(factors, assets)
        return result

    def test_beta_matches_to_long(self, setup_result):
        """Test that get_beta matches to_long."""
        result = setup_result
        beta_wide = result.get_beta("f1")
        beta_long = result.to_long("f1")

        # Stack the wide format and compare
        beta_stacked = beta_wide.stack().reset_index()
        beta_stacked.columns = ["date", "asset", "beta"]

        pd.testing.assert_frame_equal(
            beta_stacked[["beta"]].reset_index(drop=True),
            beta_long[["beta"]].reset_index(drop=True),
            check_dtype=False,
        )

    def test_result_preserves_index(self, setup_result):
        """Test that result preserves original index."""
        result = setup_result
        assert result.index is not None
        assert len(result.index) == 100

    def test_result_preserves_columns(self, setup_result):
        """Test that result preserves original column names."""
        result = setup_result
        assert list(result.factor_cols) == ["f1", "f2"]
        assert list(result.asset_cols) == ["a1", "a2"]


class TestRollingOLSResultRanges:
    """Tests for value ranges and constraints."""

    @pytest.fixture
    def setup_result(self):
        """Create a result for range testing."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 1), columns=["f1"])
        assets = pd.DataFrame(np.random.randn(T, 1), columns=["a1"])

        ols = RollingOLS(window=20, adj_r2=False)
        result = ols.fit_transform(factors, assets)
        return result

    def test_r2_in_valid_range(self, setup_result):
        """Test that R² values are in [0, 1]."""
        result = setup_result
        r2 = result.get_r2("f1").values.flatten()
        r2_valid = r2[~np.isnan(r2)]

        if len(r2_valid) > 0:
            assert (r2_valid >= 0).all() and (r2_valid <= 1).all()

    def test_betas_not_all_nan(self, setup_result):
        """Test that betas are not all NaN."""
        result = setup_result
        beta = result.get_beta("f1").values.flatten()
        beta_valid = beta[~np.isnan(beta)]

        assert len(beta_valid) > 0

    def test_signals_same_shape_as_betas(self, setup_result):
        """Test that signals have same shape as betas."""
        result = setup_result
        beta = result.get_beta("f1")
        signal = result.get_signal("f1")

        assert beta.shape == signal.shape
