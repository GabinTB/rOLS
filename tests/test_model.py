"""Tests for the RollingOLS model class."""

import numpy as np
import pandas as pd
import pytest

from rols.estimators import rolling_joint_solve, rolling_residualize
from rols.model import RollingOLS
from tests.oracle import oracle_rolling


class TestRollingOLSInit:
    """Tests for RollingOLS initialization."""

    def test_default_initialization(self):
        """Test default parameter initialization."""
        ols = RollingOLS()
        assert ols.window == 252
        assert ols.min_periods == 252
        assert ols.expanding is False
        assert ols.fit_intercept is True
        assert ols.lambda_ == 0.0
        assert ols.penalize_controls is True
        assert ols.adj_r2 is False
        assert ols.lag_signal is False
        assert ols.hac_lags is None

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        ols = RollingOLS(
            window=100,
            min_periods=50,
            expanding=True,
            fit_intercept=False,
            lambda_=0.01,
            penalize_controls=False,
            adj_r2=True,
            lag_signal=True,
            hac_lags=5,
        )
        assert ols.window == 100
        assert ols.min_periods == 50
        assert ols.expanding is True
        assert ols.fit_intercept is False
        assert ols.lambda_ == 0.01
        assert ols.penalize_controls is False
        assert ols.adj_r2 is True
        assert ols.lag_signal is True
        assert ols.hac_lags == 5

    def test_min_periods_defaults_to_window(self):
        """Test that min_periods defaults to window."""
        ols = RollingOLS(window=100)
        assert ols.min_periods == 100

    def test_negative_lambda_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            RollingOLS(lambda_=-0.1)


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


class TestRollingOLSIndexContract:
    """Indexes must be identical, unique, and increasing across all inputs."""

    @staticmethod
    def _frames(index: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        size = len(index)
        positions = np.arange(size, dtype=float)
        factors = pd.DataFrame({"factor": positions}, index=index)
        controls = pd.DataFrame({"control": positions**2}, index=index)
        assets = pd.DataFrame({"asset": np.sin(positions)}, index=index)
        return factors, controls, assets

    def test_identical_indexes_are_accepted(self):
        index = pd.date_range("2024-01-01", periods=8)
        factors, controls, assets = self._frames(index)

        result = RollingOLS(window=4).fit_transform(factors, assets, controls=controls)

        assert result.index.equals(index)

    def test_permuted_control_index_raises_and_names_frames(self):
        index = pd.date_range("2024-01-01", periods=8)
        factors, controls, _ = self._frames(index)
        controls = controls.iloc[::-1]

        with pytest.raises(ValueError, match="'factors'.*'controls'"):
            RollingOLS(window=4).fit(factors, controls=controls)

    def test_partially_overlapping_index_raises_at_first_divergence(self):
        factor_index = pd.date_range("2024-01-01", periods=8)
        control_index = pd.date_range("2024-01-02", periods=8)
        factors, _, _ = self._frames(factor_index)
        _, controls, _ = self._frames(control_index)

        with pytest.raises(ValueError, match="differ from position 0"):
            RollingOLS(window=4).fit(factors, controls=controls)

    def test_duplicate_index_raises_and_names_labels(self):
        index = pd.Index([0, 1, 1, 2])
        factors, _, _ = self._frames(index)

        with pytest.raises(ValueError, match=r"'factors'.*duplicate labels \[1\]"):
            RollingOLS(window=2).fit(factors)

    def test_non_monotonic_index_raises(self):
        index = pd.Index([0, 2, 1, 3])
        factors, _, _ = self._frames(index)

        with pytest.raises(ValueError, match="'factors'.*not monotonically increasing"):
            RollingOLS(window=2).fit(factors)

    def test_different_lengths_raise_and_report_lengths(self):
        factor_index = pd.RangeIndex(8)
        control_index = pd.RangeIndex(7)
        factors, _, _ = self._frames(factor_index)
        _, controls, _ = self._frames(control_index)

        with pytest.raises(ValueError, match="lengths 8 and 7"):
            RollingOLS(window=4).fit(factors, controls=controls)

    def test_different_index_types_raise(self):
        factors, _, _ = self._frames(pd.RangeIndex(8))
        _, controls, _ = self._frames(pd.Index(np.arange(8)))

        with pytest.raises(ValueError, match="different index types"):
            RollingOLS(window=4).fit(factors, controls=controls)

    def test_transform_index_must_match_fitted_index(self):
        factor_index = pd.date_range("2024-01-01", periods=8)
        asset_index = pd.date_range("2024-01-02", periods=8)
        factors, _, _ = self._frames(factor_index)
        _, _, assets = self._frames(asset_index)
        model = RollingOLS(window=4).fit(factors)

        with pytest.raises(ValueError, match="'factors'.*'assets'.*differ from position 0"):
            model.transform(assets)

    def test_permuted_target_index_raises_instead_of_returning_corrupt_result(self):
        index = pd.date_range("2024-01-01", periods=12)
        factors, _, assets = self._frames(index)
        model = RollingOLS(window=4).fit(factors)
        model.transform(assets)
        permuted_assets = assets.iloc[::-1]

        with pytest.raises(ValueError, match="'factors'.*'assets'"):
            model.transform(permuted_assets)

    def test_numpy_boundary_asserts_identical_indexes(self):
        y = pd.DataFrame({"asset": np.arange(6.0)}, index=pd.RangeIndex(6))
        X = pd.DataFrame({"factor": np.arange(6.0)}, index=pd.RangeIndex(1, 7))

        with pytest.raises(AssertionError, match="identical indexes"):
            rolling_residualize(y, X, window=3, min_periods=3, expanding=False)


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

    def test_exact_intercept_model_reports_one_coherent_fit(self):
        index = pd.RangeIndex(30)
        factors = pd.DataFrame({"factor": np.linspace(-2.0, 3.0, len(index))}, index=index)
        assets = pd.DataFrame({"asset": 3.0 + 2.0 * factors["factor"]}, index=index)

        result = RollingOLS(window=12, min_periods=8, dtype="float64").fit_transform(
            factors, assets
        )

        np.testing.assert_allclose(result.get_intercept("factor").iloc[7:], 3.0, atol=1e-12)
        np.testing.assert_allclose(result.get_beta("factor").iloc[7:], 2.0, atol=1e-12)
        np.testing.assert_allclose(result.get_residuals("factor").iloc[7:], 0.0, atol=1e-12)
        np.testing.assert_allclose(result.get_r2("factor").iloc[7:], 1.0, atol=1e-12)

    @pytest.mark.parametrize(
        ("n_controls", "fit_intercept", "expanding"),
        [
            (0, True, False),
            (1, True, False),
            (3, True, False),
            (1, False, False),
            (1, True, True),
            (1, False, True),
        ],
    )
    def test_joint_model_matches_scalar_oracle(
        self,
        panel_factory,
        n_controls,
        fit_intercept,
        expanding,
    ):
        targets, factors, controls = panel_factory(
            n_observations=60,
            n_targets=2,
            n_factors=1,
            n_controls=n_controls,
            correlation=0.7,
            nonzero_means=True,
            seed=13,
        )
        model = RollingOLS(
            window=20,
            min_periods=10,
            expanding=expanding,
            fit_intercept=fit_intercept,
            dtype="float64",
        )
        result = model.fit_transform(factors, targets, controls=controls)
        expected = oracle_rolling(
            targets,
            factors,
            controls,
            window=20,
            min_periods=10,
            expanding=expanding,
            fit_intercept=fit_intercept,
        )
        factor = factors.columns[0]

        comparisons = {
            "beta": result.get_beta(factor),
            "intercept": result.get_intercept(factor),
            "residuals": result.get_residuals(factor),
            "r2": result.get_r2(factor),
            "dof": result.get_dof(factor),
            "n_used": result.get_n_used(factor),
        }
        for quantity, actual in comparisons.items():
            np.testing.assert_allclose(
                actual,
                expected[quantity][factor],
                rtol=1e-9,
                atol=1e-12,
                equal_nan=True,
            )

    def test_controls_have_single_warmup(self):
        rng = np.random.default_rng(14)
        n_observations = 600
        index = pd.RangeIndex(n_observations)
        factors = pd.DataFrame({"factor": rng.normal(size=n_observations)}, index=index)
        controls = pd.DataFrame({"control": rng.normal(size=n_observations)}, index=index)
        assets = pd.DataFrame({"asset": rng.normal(size=n_observations)}, index=index)

        result = RollingOLS(window=252, min_periods=252, dtype="float64").fit_transform(
            factors, assets, controls=controls
        )
        first_valid = np.flatnonzero(result.get_beta("factor").notna().to_numpy()[:, 0])[0]

        assert first_valid == 251

    def test_nonzero_mean_controls_change_the_old_nested_composition(self):
        rng = np.random.default_rng(15)
        n_observations = 100
        factor_values = 2.0 + rng.normal(size=n_observations)
        control_values = 1.0 + 0.8 * factor_values + rng.normal(scale=0.2, size=n_observations)
        factors = pd.DataFrame({"factor": factor_values})
        controls = pd.DataFrame({"control": control_values})
        assets = pd.DataFrame(
            {"asset": 5.0 + 2.0 * factor_values - 3.0 * control_values + rng.normal(size=100)}
        )

        current = RollingOLS(window=20, min_periods=20, dtype="float64").fit_transform(
            factors, assets, controls=controls
        )
        old_factor_residual = rolling_residualize(
            factors, controls, window=20, min_periods=20, expanding=False
        )["factor"]
        old_asset_residual = rolling_residualize(
            assets, controls, window=20, min_periods=20, expanding=False
        )
        old_beta = (
            old_asset_residual.rolling(20, min_periods=20)
            .cov(old_factor_residual)
            .div(old_factor_residual.rolling(20, min_periods=20).var(), axis=0)
        )
        comparison_rows = current.get_beta("factor").notna() & old_beta.notna()

        assert not np.allclose(
            current.get_beta("factor").to_numpy()[comparison_rows.to_numpy()],
            old_beta.to_numpy()[comparison_rows.to_numpy()],
        )

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


class TestRollingOLSRidge:
    """Ridge uses one normalized, standardized joint window solve."""

    @pytest.mark.parametrize("n_observations", [20, 80])
    def test_closed_form_is_stable_across_window_lengths(self, n_observations):
        rng = np.random.default_rng(n_observations)
        factor_values = rng.normal(size=n_observations)
        factor_values = (factor_values - factor_values.mean()) / factor_values.std(ddof=0)
        factors = pd.DataFrame({"factor": factor_values})
        assets = pd.DataFrame({"asset": 3.0 + 2.0 * factor_values})
        lambda_ = 0.25

        result = RollingOLS(
            window=n_observations,
            lambda_=lambda_,
            dtype="float64",
        ).fit_transform(factors, assets)

        assert result.get_beta("factor").iloc[-1, 0] == pytest.approx(
            2.0 / (1.0 + lambda_), abs=1e-12
        )
        assert result.get_intercept("factor").iloc[-1, 0] == pytest.approx(3.0, abs=1e-12)

    @pytest.mark.parametrize("n_controls", [1, 3])
    @pytest.mark.parametrize("penalize_controls", [True, False])
    @pytest.mark.parametrize("fit_intercept", [True, False])
    @pytest.mark.parametrize("ewma_halflife", [None, 7])
    def test_joint_ridge_with_controls_matches_oracle(
        self,
        panel_factory,
        n_controls,
        penalize_controls,
        fit_intercept,
        ewma_halflife,
    ):
        targets, factors, controls = panel_factory(
            n_observations=70,
            n_targets=2,
            n_factors=1,
            n_controls=n_controls,
            correlation=0.7,
            nonzero_means=True,
            seed=21,
        )
        lambda_ = 0.4
        result = RollingOLS(
            window=25,
            min_periods=15,
            lambda_=lambda_,
            penalize_controls=penalize_controls,
            fit_intercept=fit_intercept,
            ewma_halflife=ewma_halflife,
            dtype="float64",
        ).fit_transform(factors, targets, controls=controls, return_control_betas=True)
        expected = oracle_rolling(
            targets,
            factors,
            controls,
            window=25,
            min_periods=15,
            expanding=False,
            fit_intercept=fit_intercept,
            lambda_=lambda_,
            ewma_halflife=ewma_halflife,
            penalize_controls=penalize_controls,
        )
        factor = factors.columns[0]

        comparisons = {
            "beta": result.get_beta(factor),
            "intercept": result.get_intercept(factor),
            "residuals": result.get_residuals(factor),
            "r2": result.get_r2(factor),
        }
        for quantity, actual in comparisons.items():
            np.testing.assert_allclose(
                actual,
                expected[quantity][factor],
                rtol=1e-9,
                atol=1e-12,
                equal_nan=True,
            )
        for control in controls.columns:
            np.testing.assert_allclose(
                result.get_control_beta(factor, control),
                expected["control_beta"][factor][control],
                rtol=1e-9,
                atol=1e-12,
                equal_nan=True,
            )

    def test_large_penalty_shrinks_slopes_not_intercept(self):
        rng = np.random.default_rng(22)
        factor_values = rng.normal(loc=4.0, size=60)
        factors = pd.DataFrame({"factor": factor_values})
        assets = pd.DataFrame({"asset": 7.0 + 2.0 * factor_values})

        result = RollingOLS(window=60, lambda_=1e10, dtype="float64").fit_transform(factors, assets)

        assert abs(result.get_beta("factor").iloc[-1, 0]) < 1e-8
        assert result.get_intercept("factor").iloc[-1, 0] == pytest.approx(
            assets["asset"].mean(), abs=1e-8
        )

    def test_scale_invariance(self):
        rng = np.random.default_rng(23)
        factors = pd.DataFrame(rng.normal(size=(80, 2)), columns=["f1", "f2"])
        assets = pd.DataFrame(
            {"asset": 2.0 + 1.5 * factors["f1"] - 0.75 * factors["f2"] + rng.normal(size=80)}
        )
        scaled_factors = factors.copy()
        scaled_factors["f1"] *= 1000.0

        baseline = RollingOLS(window=30, lambda_=0.6, dtype="float64").fit_transform(
            factors, assets
        )
        scaled = RollingOLS(window=30, lambda_=0.6, dtype="float64").fit_transform(
            scaled_factors, assets
        )

        np.testing.assert_allclose(
            scaled.get_beta("f1") * 1000.0,
            baseline.get_beta("f1"),
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            scaled.get_beta("f2"),
            baseline.get_beta("f2"),
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        )

    def test_uniform_weights_reproduce_unweighted_ridge(self):
        rng = np.random.default_rng(24)
        factors = pd.DataFrame(rng.normal(size=(50, 2)), columns=["f1", "f2"])
        assets = pd.DataFrame(rng.normal(size=(50, 2)), columns=["a1", "a2"])
        penalty = np.diag([0.0, 0.5, 0.5])

        unweighted = rolling_joint_solve(assets, factors, 20, 10, False, penalty=penalty)
        uniform = rolling_joint_solve(
            assets,
            factors,
            20,
            10,
            False,
            penalty=penalty,
            weights=np.ones(20),
        )

        np.testing.assert_allclose(uniform.coef, unweighted.coef, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(
            uniform.intercept, unweighted.intercept, atol=1e-12, equal_nan=True
        )

    def test_zero_penalty_reproduces_ols(self):
        rng = np.random.default_rng(25)
        factors = pd.DataFrame(rng.normal(size=(50, 2)), columns=["f1", "f2"])
        assets = pd.DataFrame(rng.normal(size=(50, 2)), columns=["a1", "a2"])

        ols = rolling_joint_solve(assets, factors, 20, 10, False)
        zero_penalty = rolling_joint_solve(assets, factors, 20, 10, False, penalty=np.zeros((3, 3)))

        np.testing.assert_allclose(zero_penalty.coef, ols.coef, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(
            zero_penalty.intercept, ols.intercept, atol=1e-12, equal_nan=True
        )

    def test_joint_ridge_differs_from_penalized_fwl_hybrid(self):
        rng = np.random.default_rng(26)
        factor_values = 2.0 + rng.normal(size=100)
        control_values = 1.0 + 0.8 * factor_values + rng.normal(scale=0.2, size=100)
        factors = pd.DataFrame({"factor": factor_values})
        controls = pd.DataFrame({"control": control_values})
        assets = pd.DataFrame(
            {"asset": 5.0 + 2.0 * factor_values - 3.0 * control_values + rng.normal(size=100)}
        )
        lambda_ = 0.5

        direct = (
            RollingOLS(window=20, lambda_=lambda_, dtype="float64")
            .fit_transform(factors, assets, controls=controls)
            .get_beta("factor")
        )
        factor_residual = rolling_residualize(
            factors,
            controls,
            window=20,
            min_periods=20,
            expanding=False,
            ridge_lambda=lambda_,
        )["factor"]
        asset_residual = rolling_residualize(
            assets,
            controls,
            window=20,
            min_periods=20,
            expanding=False,
            ridge_lambda=lambda_,
        )
        hybrid = (
            asset_residual.rolling(20, min_periods=20)
            .cov(factor_residual)
            .div(factor_residual.rolling(20, min_periods=20).var(), axis=0)
        )
        valid = direct.notna() & hybrid.notna()

        assert not np.allclose(direct.to_numpy()[valid], hybrid.to_numpy()[valid])


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
