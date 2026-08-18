"""Tests for the RollingOLSResult class."""

import numpy as np
import pandas as pd
import pytest

from rols.model import RollingOLS


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

    def test_get_intercept(self, setup_result):
        """Test get_intercept method."""
        intercept = setup_result.get_intercept("f1")

        assert isinstance(intercept, pd.DataFrame)
        assert intercept.shape == (100, 3)

    def test_get_dof_and_n_used(self, setup_result):
        """Test fit sample metadata getters."""
        dof = setup_result.get_dof("f1")
        n_used = setup_result.get_n_used("f1")

        assert dof.shape == (100, 3)
        assert n_used.shape == (100, 3)
        np.testing.assert_allclose(dof.dropna(), n_used.dropna() - 2.0)

    def test_get_signal(self, setup_result):
        """Test get_signal method."""
        result = setup_result
        signal = result.get_signal("f1")

        assert isinstance(signal, pd.DataFrame)
        assert signal.shape == (100, 3)

    def test_removed_raw_exposure_signal_accessor_is_absent(self, setup_result):
        assert not hasattr(setup_result, "get_raw_exposure_signal")

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

    def test_get_partial_r2(self, setup_result):
        """Partial R² is exposed separately from full-model R²."""
        partial_r2 = setup_result.get_partial_r2("f1")

        assert isinstance(partial_r2, pd.DataFrame)
        assert partial_r2.shape == (100, 3)

    def test_get_partial_r2_invalid_factor_raises(self, setup_result):
        with pytest.raises(KeyError):
            setup_result.get_partial_r2("invalid_factor")

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
        assert not np.isinf(tstat.to_numpy()).any()

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


class TestRollingOLSResultFactorAdjustedReturns:
    """Tests for get_factor_adjusted_returns()."""

    def _data(self, T=120):
        np.random.seed(42)
        factors = pd.DataFrame(np.random.randn(T, 2), columns=["f1", "f2"])
        controls = pd.DataFrame(np.random.randn(T, 2), columns=["c1", "c2"])
        assets = pd.DataFrame(np.random.randn(T, 3), columns=["a1", "a2", "a3"])
        return factors, controls, assets

    def test_shape_with_controls(self):
        """Returns a DataFrame of shape (T, N_assets) when controls are present."""
        factors, controls, assets = self._data()
        ols = RollingOLS(window=20)
        result = ols.fit_transform(factors, assets, controls=controls)

        far = result.get_factor_adjusted_returns()
        assert isinstance(far, pd.DataFrame)
        assert far.shape == (120, 3)
        assert list(far.columns) == ["a1", "a2", "a3"]

    def test_differs_from_original_with_controls(self):
        """With controls, factor-adjusted returns differ from raw asset returns."""
        factors, controls, assets = self._data()
        ols = RollingOLS(window=20)
        result = ols.fit_transform(factors, assets, controls=controls)

        far = result.get_factor_adjusted_returns()
        # Where both are defined, residualized values should not equal raw returns.
        assert not np.allclose(
            far.dropna().values,
            assets.astype(far.dtypes.iloc[0]).loc[far.dropna().index].values,
        )

    def test_equals_original_without_controls(self):
        """Without controls, factor-adjusted returns equal the original returns."""
        factors, _, assets = self._data()
        ols = RollingOLS(window=20)
        result = ols.fit_transform(factors, assets)

        far = result.get_factor_adjusted_returns()
        pd.testing.assert_frame_equal(far, assets.astype(far.dtypes.iloc[0]), check_dtype=False)

    def test_differs_from_regression_residuals(self):
        """FWL step 2 (controls only) differs from get_residuals (step 3)."""
        factors, controls, assets = self._data()
        ols = RollingOLS(window=20)
        result = ols.fit_transform(factors, assets, controls=controls)

        far = result.get_factor_adjusted_returns()
        resids = result.get_residuals("f1")
        assert far.shape == resids.shape
        # The two quantities are different — step 3 removes the factor too.
        mask = far.notna() & resids.notna()
        assert not np.allclose(far.values[mask.values], resids.values[mask.values])


class TestRollingOLSResultControlBetas:
    """Tests for get_control_beta()."""

    @pytest.fixture
    def setup_with_control_betas(self):
        """Result computed with return_control_betas=True and two controls."""
        np.random.seed(42)
        T = 120
        factors = pd.DataFrame(np.random.randn(T, 2), columns=["f1", "f2"])
        controls = pd.DataFrame(np.random.randn(T, 2), columns=["c1", "c2"])
        assets = pd.DataFrame(np.random.randn(T, 3), columns=["a1", "a2", "a3"])

        ols = RollingOLS(window=20)
        return ols.fit_transform(factors, assets, controls=controls, return_control_betas=True)

    def test_get_control_beta_shape(self, setup_with_control_betas):
        """get_control_beta returns shape (T, N_assets)."""
        result = setup_with_control_betas
        cb = result.get_control_beta("f1", "c1")

        assert isinstance(cb, pd.DataFrame)
        assert cb.shape == (120, 3)
        assert list(cb.columns) == ["a1", "a2", "a3"]

    def test_get_control_beta_invalid_factor_raises(self, setup_with_control_betas):
        """Unknown factor raises KeyError."""
        result = setup_with_control_betas
        with pytest.raises(KeyError):
            result.get_control_beta("nope", "c1")

    def test_get_control_beta_invalid_control_raises(self, setup_with_control_betas):
        """Unknown control raises KeyError."""
        result = setup_with_control_betas
        with pytest.raises(KeyError):
            result.get_control_beta("f1", "nope")

    def test_get_control_beta_default_raises(self):
        """Default (return_control_betas=False) raises RuntimeError."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 1), columns=["f1"])
        controls = pd.DataFrame(np.random.randn(T, 1), columns=["c1"])
        assets = pd.DataFrame(np.random.randn(T, 2), columns=["a1", "a2"])

        ols = RollingOLS(window=20)
        result = ols.fit_transform(factors, assets, controls=controls)

        with pytest.raises(RuntimeError):
            result.get_control_beta("f1", "c1")

    def test_get_control_beta_no_controls_raises(self):
        """return_control_betas=True without controls raises RuntimeError."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 1), columns=["f1"])
        assets = pd.DataFrame(np.random.randn(T, 2), columns=["a1", "a2"])

        ols = RollingOLS(window=20)
        result = ols.fit_transform(factors, assets, return_control_betas=True)

        with pytest.raises(RuntimeError):
            result.get_control_beta("f1", "c1")


class TestFactorMimickingReturnsRemoved:
    """Guard that the removed accessors cannot silently regress into the API."""

    def test_get_factor_mimicking_returns_absent(self):
        """get_factor_mimicking_returns must not exist on RollingOLSResult."""
        from rols.results import RollingOLSResult

        assert not hasattr(RollingOLSResult, "get_factor_mimicking_returns"), (
            "get_factor_mimicking_returns was removed in v0.3.0 (F13). "
            "Do not add it back without a proper cross-sectional specification."
        )

    def test_get_all_factor_mimicking_returns_absent(self):
        """get_all_factor_mimicking_returns must not exist on RollingOLSResult."""
        from rols.results import RollingOLSResult

        assert not hasattr(RollingOLSResult, "get_all_factor_mimicking_returns"), (
            "get_all_factor_mimicking_returns was removed in v0.3.0 (F13). "
            "Do not add it back without a proper cross-sectional specification."
        )

    def test_readme_has_no_fama_macbeth_claim(self):
        """README must not mention Fama-MacBeth or cross-sectional estimation."""
        import pathlib

        readme = pathlib.Path(__file__).parent.parent / "README.md"
        text = readme.read_text(encoding="utf-8").lower()
        forbidden = ["fama-macbeth", "fama macbeth", "cross-sectional estimation"]
        for phrase in forbidden:
            assert phrase not in text, (
                f"README contains '{phrase}'. "
                "Cross-sectional estimation is out of scope — see CHANGELOG."
            )

    def test_readme_has_no_factor_mimicking_claim(self):
        """README must not mention factor-mimicking returns."""
        import pathlib

        readme = pathlib.Path(__file__).parent.parent / "README.md"
        text = readme.read_text(encoding="utf-8").lower()
        assert "factor mimicking" not in text, (
            "README contains 'factor mimicking'. "
            "The accessors were removed in v0.3.0 — do not re-document them."
        )
