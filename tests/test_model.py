"""Tests for the RollingOLS model class."""

import numpy as np
import pandas as pd
import pytest

from rols.model import RollingOLS


class TestRollingOLSInit:
    """Tests for RollingOLS initialization."""

    def test_default_initialization(self):
        """Test default parameter initialization."""
        ols = RollingOLS()
        assert ols.window == 252
        assert ols.min_periods == 252
        assert ols.expanding is False
        assert ols.lambda_ == 0.0
        assert ols.adj_r2 is False
        assert ols.lag_signal is False
        assert ols.hac_lags is None

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        ols = RollingOLS(
            window=100,
            min_periods=50,
            expanding=True,
            lambda_=0.01,
            adj_r2=True,
            lag_signal=True,
            hac_lags=5,
        )
        assert ols.window == 100
        assert ols.min_periods == 50
        assert ols.expanding is True
        assert ols.lambda_ == 0.01
        assert ols.adj_r2 is True
        assert ols.lag_signal is True
        assert ols.hac_lags == 5

    def test_min_periods_defaults_to_window(self):
        """Test that min_periods defaults to window."""
        ols = RollingOLS(window=100)
        assert ols.min_periods == 100


class TestRollingOLSFit:
    """Tests for the fit() method."""

    def test_basic_fit(self):
        """Test basic fit with factors."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 2), columns=["f1", "f2"])

        ols = RollingOLS(window=20)
        ols.fit(factors)

        assert ols._is_fitted
        assert list(ols._factor_cols) == ["f1", "f2"]
        assert ols._control_cols == []

    def test_fit_with_controls(self):
        """Test fit with control variables."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 2), columns=["f1", "f2"])
        controls = pd.DataFrame(np.random.randn(T, 2), columns=["c1", "c2"])

        ols = RollingOLS(window=20)
        ols.fit(factors, controls=controls)

        assert list(ols._control_cols) == ["c1", "c2"]
        assert ols._controls_fitted is not None

    def test_fit_returns_self(self):
        """Test that fit returns self for chaining."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 1), columns=["f1"])

        ols = RollingOLS(window=20)
        result = ols.fit(factors)

        assert result is ols

    def test_fit_orthogonalize_factors(self):
        """Test fit with factor orthogonalization."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 3), columns=["f1", "f2", "f3"])

        ols = RollingOLS(window=20)
        ols.fit(factors, orthogonalize_factors=True)

        assert ols._is_fitted

    def test_fit_orthogonalize_controls(self):
        """Test fit with control orthogonalization."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 1), columns=["f1"])
        controls = pd.DataFrame(np.random.randn(T, 2), columns=["c1", "c2"])

        ols = RollingOLS(window=20)
        ols.fit(factors, controls=controls, orthogonalize_controls=True)

        assert ols._is_fitted

    def test_fit_with_ridge(self):
        """Test fit with ridge regularization."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 2), columns=["f1", "f2"])

        ols = RollingOLS(window=20, lambda_=0.01)
        ols.fit(factors)

        assert ols.lambda_ == 0.01


class TestRollingOLSTransform:
    """Tests for the transform() method."""

    def test_basic_transform(self):
        """Test basic transform."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 2), columns=["f1", "f2"])
        assets = pd.DataFrame(np.random.randn(T, 3), columns=["a1", "a2", "a3"])

        ols = RollingOLS(window=20)
        ols.fit(factors)
        result = ols.transform(assets)

        assert result is not None
        assert list(result.factor_cols) == ["f1", "f2"]
        assert list(result.asset_cols) == ["a1", "a2", "a3"]

    def test_transform_before_fit_raises(self):
        """Test that transform before fit raises error."""
        np.random.seed(42)
        T = 100
        assets = pd.DataFrame(np.random.randn(T, 1), columns=["a1"])

        ols = RollingOLS(window=20)

        with pytest.raises(RuntimeError):
            ols.transform(assets)

    def test_transform_result_shapes(self):
        """Test that transform result has correct shapes."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 2), columns=["f1", "f2"])
        assets = pd.DataFrame(np.random.randn(T, 3), columns=["a1", "a2", "a3"])

        ols = RollingOLS(window=20)
        ols.fit(factors)
        result = ols.transform(assets)

        # Beta shape should be (T, N_assets) for each factor
        for factor in ["f1", "f2"]:
            beta = result.get_beta(factor)
            assert beta.shape == (T, 3)
            assert list(beta.columns) == ["a1", "a2", "a3"]

    def test_transform_with_controls(self):
        """Test transform with control variables."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 1), columns=["f1"])
        controls = pd.DataFrame(np.random.randn(T, 1), columns=["c1"])
        assets = pd.DataFrame(np.random.randn(T, 2), columns=["a1", "a2"])

        ols = RollingOLS(window=20)
        ols.fit(factors, controls=controls)
        result = ols.transform(assets)

        assert result is not None

    def test_transform_with_nan_assets(self):
        """Test transform with NaN values in assets."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 1), columns=["f1"])
        assets = pd.DataFrame(np.random.randn(T, 2), columns=["a1", "a2"])
        assets.iloc[10:15] = np.nan

        ols = RollingOLS(window=20)
        ols.fit(factors)
        result = ols.transform(assets)

        assert result is not None

    def test_control_beta_shape(self):
        """get_control_beta returns shape (T, N_assets)."""
        np.random.seed(42)
        T = 120
        factors = pd.DataFrame(np.random.randn(T, 2), columns=["f1", "f2"])
        controls = pd.DataFrame(np.random.randn(T, 2), columns=["c1", "c2"])
        assets = pd.DataFrame(np.random.randn(T, 3), columns=["a1", "a2", "a3"])

        ols = RollingOLS(window=20)
        result = ols.fit_transform(factors, assets, controls=controls, return_control_betas=True)

        cb = result.get_control_beta("f1", "c1")
        assert cb.shape == (T, 3)
        assert list(cb.columns) == ["a1", "a2", "a3"]

    def test_control_beta_shared_across_factors(self):
        """Control betas do not depend on the factor — identical for all factors."""
        np.random.seed(0)
        T = 120
        factors = pd.DataFrame(np.random.randn(T, 2), columns=["f1", "f2"])
        controls = pd.DataFrame(np.random.randn(T, 2), columns=["c1", "c2"])
        assets = pd.DataFrame(np.random.randn(T, 2), columns=["a1", "a2"])

        ols = RollingOLS(window=20)
        result = ols.fit_transform(factors, assets, controls=controls, return_control_betas=True)

        pd.testing.assert_frame_equal(
            result.get_control_beta("f1", "c1"),
            result.get_control_beta("f2", "c1"),
        )

    def test_control_beta_single_control_matches_univariate(self):
        """With one control, FWL is identity: control beta == plain univariate beta."""
        np.random.seed(1)
        T = 150
        factors = pd.DataFrame(np.random.randn(T, 1), columns=["f1"])
        controls = pd.DataFrame(np.random.randn(T, 1), columns=["c1"])
        assets = pd.DataFrame(np.random.randn(T, 2), columns=["a1", "a2"])

        ols = RollingOLS(window=30)
        result = ols.fit_transform(factors, assets, controls=controls, return_control_betas=True)
        joint = result.get_control_beta("f1", "c1")

        # Plain rolling univariate beta of assets on the single control
        c = controls["c1"]
        cov = assets.rolling(30, min_periods=30).cov(c)
        var = c.rolling(30, min_periods=30).var()
        univariate = cov.div(var, axis=0)

        pd.testing.assert_frame_equal(
            joint, univariate.astype(joint.dtypes), check_dtype=False, rtol=1e-3
        )

    def test_control_beta_multiple_correlated_differs_from_univariate(self):
        """With correlated controls, joint beta differs from the marginal univariate beta."""
        np.random.seed(7)
        T = 200
        # c2 is strongly correlated with c1
        c1 = np.random.randn(T)
        c2 = 0.9 * c1 + 0.1 * np.random.randn(T)
        controls = pd.DataFrame({"c1": c1, "c2": c2})
        factors = pd.DataFrame(np.random.randn(T, 1), columns=["f1"])
        assets = pd.DataFrame(np.random.randn(T, 1), columns=["a1"])

        ols = RollingOLS(window=40)
        result = ols.fit_transform(factors, assets, controls=controls, return_control_betas=True)
        joint = result.get_control_beta("f1", "c1")

        # Marginal univariate beta ignoring c2
        cov = assets.rolling(40, min_periods=40).cov(controls["c1"])
        var = controls["c1"].rolling(40, min_periods=40).var()
        univariate = cov.div(var, axis=0)

        # Compare on rows where both are defined; they should meaningfully differ
        mask = joint["a1"].notna() & univariate["a1"].notna()
        diff = (joint["a1"][mask] - univariate["a1"][mask]).abs()
        assert diff.mean() > 1e-3

    def test_control_beta_default_raises(self):
        """Default return_control_betas=False: get_control_beta raises RuntimeError."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 1), columns=["f1"])
        controls = pd.DataFrame(np.random.randn(T, 1), columns=["c1"])
        assets = pd.DataFrame(np.random.randn(T, 2), columns=["a1", "a2"])

        ols = RollingOLS(window=20)
        result = ols.fit_transform(factors, assets, controls=controls)

        with pytest.raises(RuntimeError):
            result.get_control_beta("f1", "c1")

    def test_control_beta_no_controls_raises(self):
        """return_control_betas=True but no controls passed: get_control_beta raises."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 1), columns=["f1"])
        assets = pd.DataFrame(np.random.randn(T, 2), columns=["a1", "a2"])

        ols = RollingOLS(window=20)
        result = ols.fit_transform(factors, assets, return_control_betas=True)

        with pytest.raises(RuntimeError):
            result.get_control_beta("f1", "c1")


class TestRollingOLSFitTransform:
    """Tests for fit_transform() method."""

    def test_basic_fit_transform(self):
        """Test basic fit_transform."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 2), columns=["f1", "f2"])
        assets = pd.DataFrame(np.random.randn(T, 2), columns=["a1", "a2"])

        ols = RollingOLS(window=20)
        result = ols.fit_transform(factors, assets)

        assert result is not None
        assert list(result.factor_cols) == ["f1", "f2"]
        assert list(result.asset_cols) == ["a1", "a2"]

    def test_fit_transform_with_controls(self):
        """Test fit_transform with controls."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 1), columns=["f1"])
        controls = pd.DataFrame(np.random.randn(T, 1), columns=["c1"])
        assets = pd.DataFrame(np.random.randn(T, 2), columns=["a1", "a2"])

        ols = RollingOLS(window=20)
        result = ols.fit_transform(factors, assets, controls=controls)

        assert result is not None

    def test_fit_transform_equivalent_to_fit_then_transform(self):
        """Test that fit_transform equals fit().transform()."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 2), columns=["f1", "f2"])
        assets = pd.DataFrame(np.random.randn(T, 2), columns=["a1", "a2"])

        # fit_transform
        ols1 = RollingOLS(window=20)
        result1 = ols1.fit_transform(factors, assets)

        # fit then transform
        ols2 = RollingOLS(window=20)
        ols2.fit(factors)
        result2 = ols2.transform(assets)

        # Results should be identical
        for factor in result1.factor_cols:
            pd.testing.assert_frame_equal(
                result1.get_beta(factor), result2.get_beta(factor), check_dtype=False
            )
            pd.testing.assert_frame_equal(
                result1.get_signal(factor), result2.get_signal(factor), check_dtype=False
            )


class TestRollingOLSModes:
    """Tests for different window modes."""

    def test_rolling_mode(self):
        """Test rolling window mode."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 1), columns=["f1"])
        assets = pd.DataFrame(np.random.randn(T, 1), columns=["a1"])

        ols = RollingOLS(window=20, expanding=False)
        result = ols.fit_transform(factors, assets)

        assert result is not None

    def test_expanding_mode(self):
        """Test expanding window mode."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 1), columns=["f1"])
        assets = pd.DataFrame(np.random.randn(T, 1), columns=["a1"])

        ols = RollingOLS(window=20, expanding=True)
        result = ols.fit_transform(factors, assets)

        assert result is not None

    def test_lag_signal_mode(self):
        """Test lag_signal parameter."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 1), columns=["f1"])
        assets = pd.DataFrame(np.random.randn(T, 1), columns=["a1"])

        ols1 = RollingOLS(window=20, lag_signal=False)
        result1 = ols1.fit_transform(factors, assets)

        ols2 = RollingOLS(window=20, lag_signal=True)
        result2 = ols2.fit_transform(factors, assets)

        # Signals should differ with different lag_signal settings
        signal1 = result1.get_signal("f1")
        signal2 = result2.get_signal("f1")
        # They may differ at t=0 (lag_signal=True will have NaN at t=0)
        assert signal1.shape == signal2.shape

    def test_adjusted_r2(self):
        """Test adjusted R² computation."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 1), columns=["f1"])
        assets = pd.DataFrame(np.random.randn(T, 1), columns=["a1"])

        ols1 = RollingOLS(window=20, adj_r2=False)
        result1 = ols1.fit_transform(factors, assets)

        ols2 = RollingOLS(window=20, adj_r2=True)
        result2 = ols2.fit_transform(factors, assets)

        r2_1 = result1.get_r2("f1")
        r2_2 = result2.get_r2("f1")

        # Adjusted R² should typically be lower than R²
        # (at least for the non-NaN values)
        assert not r2_1.equals(r2_2)

    def test_adjusted_r2_small_min_periods_no_inf(self):
        """adj_r2 with min_periods=2 must not produce inf (issue #7)."""
        np.random.seed(42)
        T = 30
        factors = pd.DataFrame(np.random.randn(T, 1), columns=["f1"])
        assets = pd.DataFrame(np.random.randn(T, 2), columns=["a1", "a2"])

        ols = RollingOLS(window=10, min_periods=2, adj_r2=True)
        result = ols.fit_transform(factors, assets)
        r2 = result.get_r2("f1")

        # No inf anywhere — the divide-by-zero at n_obs == 2 must be guarded.
        assert not np.isinf(r2.values).any()

    def test_adjusted_r2_nobs_le_2_is_nan(self):
        """Windows with n_obs <= 2 yield NaN (undefined), not inf or a fixed 1.0."""
        np.random.seed(0)
        T = 30
        factors = pd.DataFrame(np.random.randn(T, 1), columns=["f1"])
        assets = pd.DataFrame(np.random.randn(T, 1), columns=["a1"])

        ols = RollingOLS(window=10, min_periods=2, adj_r2=True)
        result = ols.fit_transform(factors, assets)
        r2 = result.get_r2("f1")["a1"]

        # Rolling count of observations per window (matches model internals).
        n_obs = assets["a1"].rolling(10, min_periods=2).count()

        # n_obs <= 2 -> adjusted R² undefined -> NaN
        small = (n_obs <= 2) & n_obs.notna()
        assert small.any()  # the construction actually reaches this regime
        assert r2[small].isna().all()

        # n_obs > 2 -> finite, valid values
        big = n_obs > 2
        valid = r2[big].dropna()
        assert len(valid) > 0
        assert np.isfinite(valid).all()
        assert (valid <= 1.0).all()


class TestRollingOLSEWMA:
    """Tests for EWMA observation weighting (issue #1)."""

    def test_ewma_halflife_stored(self):
        """ewma_halflife is stored on the constructor; defaults to None."""
        assert RollingOLS().ewma_halflife is None
        assert RollingOLS(ewma_halflife=63).ewma_halflife == 63

    def test_ewma_with_expanding_raises(self):
        """ewma_halflife combined with expanding=True raises ValueError."""
        with pytest.raises(ValueError, match="expanding"):
            RollingOLS(window=20, expanding=True, ewma_halflife=10)

    def test_ewma_betas_differ_from_equal_weight(self):
        """EWMA betas differ from the equal-weight betas."""
        np.random.seed(42)
        T = 200
        factors = pd.DataFrame(np.random.randn(T, 2), columns=["f1", "f2"])
        assets = pd.DataFrame(np.random.randn(T, 3), columns=["a1", "a2", "a3"])

        equal = RollingOLS(window=60).fit_transform(factors, assets)
        ewma = RollingOLS(window=60, ewma_halflife=15).fit_transform(factors, assets)

        beta_eq = equal.get_beta("f1")
        beta_ew = ewma.get_beta("f1")
        # Same shape/index/columns, materially different values.
        assert beta_eq.shape == beta_ew.shape
        mask = beta_eq.notna() & beta_ew.notna()
        assert mask.values.any()
        diff = beta_eq.values[mask.values] - beta_ew.values[mask.values]
        assert np.abs(diff).mean() > 1e-3

    def test_ewma_none_unchanged_vs_baseline(self):
        """ewma_halflife=None is identical to the existing equal-weight path."""
        np.random.seed(7)
        T = 150
        factors = pd.DataFrame(np.random.randn(T, 2), columns=["f1", "f2"])
        controls = pd.DataFrame(np.random.randn(T, 1), columns=["c1"])
        assets = pd.DataFrame(np.random.randn(T, 3), columns=["a1", "a2", "a3"])

        baseline = RollingOLS(window=40).fit_transform(factors, assets, controls=controls)
        explicit = RollingOLS(window=40, ewma_halflife=None).fit_transform(
            factors, assets, controls=controls
        )
        for fac in ["f1", "f2"]:
            pd.testing.assert_frame_equal(baseline.get_beta(fac), explicit.get_beta(fac))
            pd.testing.assert_frame_equal(baseline.get_r2(fac), explicit.get_r2(fac))

    def test_ewma_beta_shape_and_index(self):
        """EWMA betas have correct shape, columns, and index."""
        np.random.seed(1)
        T = 120
        idx = pd.date_range("2020-01-01", periods=T, freq="D")
        factors = pd.DataFrame(np.random.randn(T, 1), columns=["f1"], index=idx)
        assets = pd.DataFrame(np.random.randn(T, 2), columns=["a1", "a2"], index=idx)

        result = RollingOLS(window=30, ewma_halflife=10).fit_transform(factors, assets)
        beta = result.get_beta("f1")
        assert beta.shape == (T, 2)
        assert list(beta.columns) == ["a1", "a2"]
        assert beta.index.equals(idx)

    def test_ewma_beta_matches_manual_weighted_regression(self):
        """EWMA beta equals the weighted univariate slope cov_w/var_w."""
        from rols.model import _ewma_weights

        np.random.seed(2)
        T, window, hl = 80, 30, 8
        factors = pd.DataFrame(np.random.randn(T, 1), columns=["f1"])
        assets = pd.DataFrame(np.random.randn(T, 1), columns=["a1"])

        result = RollingOLS(window=window, ewma_halflife=hl, dtype="float64").fit_transform(
            factors, assets
        )
        beta = result.get_beta("f1")["a1"]

        # Manual weighted slope for the final window.
        t = T - 1
        w = _ewma_weights(hl, window)
        f_w = factors["f1"].to_numpy()[t - window + 1 : t + 1]
        a_w = assets["a1"].to_numpy()[t - window + 1 : t + 1]
        fbar = (w * f_w).sum()
        abar = (w * a_w).sum()
        cov = (w * (f_w - fbar) * (a_w - abar)).sum()
        var = (w * (f_w - fbar) ** 2).sum()
        expected = cov / var

        assert beta.iloc[t] == pytest.approx(expected, rel=1e-9)

    def test_ewma_handles_nan_assets(self):
        """EWMA path tolerates NaN in assets without crashing."""
        np.random.seed(3)
        T = 120
        factors = pd.DataFrame(np.random.randn(T, 1), columns=["f1"])
        assets = pd.DataFrame(np.random.randn(T, 2), columns=["a1", "a2"])
        assets.iloc[10:15, 0] = np.nan

        result = RollingOLS(window=30, ewma_halflife=10).fit_transform(factors, assets)
        beta = result.get_beta("f1")
        # a1 is NaN at the masked prediction points but defined elsewhere.
        assert beta["a1"].iloc[10:15].isna().all()
        assert beta["a1"].notna().any()


class TestRollingOLSEdgeCases:
    """Tests for edge cases."""

    def test_single_factor_single_asset(self):
        """Test with single factor and single asset."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 1), columns=["f1"])
        assets = pd.DataFrame(np.random.randn(T, 1), columns=["a1"])

        ols = RollingOLS(window=20)
        result = ols.fit_transform(factors, assets)

        assert result.get_beta("f1").shape == (T, 1)

    def test_many_factors_many_assets(self):
        """Test with many factors and assets."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 10), columns=[f"f{i}" for i in range(10)])
        assets = pd.DataFrame(np.random.randn(T, 20), columns=[f"a{i}" for i in range(20)])

        ols = RollingOLS(window=20)
        result = ols.fit_transform(factors, assets)

        assert result.get_beta("f0").shape == (T, 20)

    def test_min_periods_less_than_window(self):
        """Test with min_periods less than window."""
        np.random.seed(42)
        T = 50
        factors = pd.DataFrame(np.random.randn(T, 1), columns=["f1"])
        assets = pd.DataFrame(np.random.randn(T, 1), columns=["a1"])

        ols = RollingOLS(window=20, min_periods=5)
        result = ols.fit_transform(factors, assets)

        assert result is not None
        # Should have values earlier than if min_periods=window
        beta = result.get_beta("f1")
        non_nan_count = beta.notna().sum().sum()
        assert non_nan_count > 0

    def test_dtype_float32(self):
        """Test with float32 dtype."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 2), columns=["f1", "f2"])
        assets = pd.DataFrame(np.random.randn(T, 2), columns=["a1", "a2"])

        ols = RollingOLS(window=20, dtype="float32")
        result = ols.fit_transform(factors, assets)

        assert result is not None

    def test_dtype_float64(self):
        """Test with float64 dtype."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 2), columns=["f1", "f2"])
        assets = pd.DataFrame(np.random.randn(T, 2), columns=["a1", "a2"])

        ols = RollingOLS(window=20, dtype="float64")
        result = ols.fit_transform(factors, assets)

        assert result is not None

    def test_asset_chunk_size(self):
        """Test with different asset chunk sizes."""
        np.random.seed(42)
        T = 100
        factors = pd.DataFrame(np.random.randn(T, 2), columns=["f1", "f2"])
        assets = pd.DataFrame(np.random.randn(T, 50), columns=[f"a{i}" for i in range(50)])

        ols1 = RollingOLS(window=20, asset_chunk_size=10)
        result1 = ols1.fit_transform(factors, assets)

        ols2 = RollingOLS(window=20, asset_chunk_size=50)
        result2 = ols2.fit_transform(factors, assets)

        # Results should be the same regardless of chunk size
        pd.testing.assert_frame_equal(
            result1.get_beta("f1"), result2.get_beta("f1"), check_dtype=False
        )


class TestRollingOLSSingularWarnings:
    """RollingOLS surfaces singular-matrix warnings (issue #12)."""

    def _collinear_setup(self, T=60):
        """Duplicate control columns -> singular residualization windows."""
        np.random.seed(42)
        c = np.random.randn(T)
        factors = pd.DataFrame(np.random.randn(T, 1), columns=["f1"])
        controls = pd.DataFrame({"c1": c, "c2": c})  # collinear
        assets = pd.DataFrame(np.random.randn(T, 2), columns=["a1", "a2"])
        return factors, assets, controls

    def test_warns_by_default(self):
        """Collinear controls trigger a RuntimeWarning during fit."""
        factors, assets, controls = self._collinear_setup()
        ols = RollingOLS(window=20)
        with pytest.warns(RuntimeWarning, match="singular"):
            ols.fit_transform(factors, assets, controls=controls)

    def test_warn_singular_false_suppresses(self, recwarn):
        """warn_singular=False on the constructor suppresses the warning."""
        factors, assets, controls = self._collinear_setup()
        ols = RollingOLS(window=20, warn_singular=False)
        ols.fit_transform(factors, assets, controls=controls)
        assert not any(issubclass(w.category, RuntimeWarning) for w in recwarn)
