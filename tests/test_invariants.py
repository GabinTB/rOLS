"""Statistical and structural invariants that must hold regardless of
implementation path (F22).

These tests check properties that follow from the model definition itself —
not agreement between two internal code paths, and not just shapes. Where a
test does compare two code paths (chunking, FWL-vs-joint, lazy-vs-eager), the
underlying claim is still an invariant a user could observe from the outside.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rols import RollingOLS
from rols.estimators import rolling_fwl_solve, rolling_joint_solve


def _random_panel(
    n_observations: int = 60,
    n_factors: int = 2,
    n_controls: int = 1,
    n_targets: int = 3,
    seed: int = 0,
    nan_gap: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    index = pd.RangeIndex(n_observations)
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
    if nan_gap:
        targets.iloc[5:8, 0] = np.nan
        factors.iloc[10:12, -1] = np.nan
    return factors, controls, targets


class TestCoherence:
    """y == intercept + X @ coef + resid on the complete-case sample."""

    @pytest.mark.parametrize("fit_intercept", [True, False])
    @pytest.mark.parametrize("expanding", [False, True])
    @pytest.mark.parametrize("nan_gap", [False, True])
    def test_fit_reconstructs_target_on_complete_case(self, fit_intercept, expanding, nan_gap):
        factors, controls, targets = _random_panel(nan_gap=nan_gap)
        design = pd.concat([controls, factors], axis=1)
        min_periods = 10 if not expanding else 8

        fit = rolling_joint_solve(
            targets,
            design,
            window=15,
            min_periods=min_periods,
            expanding=expanding,
            fit_intercept=fit_intercept,
            warn_singular=False,
        )
        fitted = fit.intercept + np.einsum("tp,tpn->tn", design.to_numpy(), fit.coef)
        reconstructed = fitted + fit.resid_endpoint
        valid = np.isfinite(reconstructed) & np.isfinite(targets.to_numpy())

        np.testing.assert_allclose(
            reconstructed[valid], targets.to_numpy()[valid], rtol=0, atol=1e-9
        )


class TestR2Reproducibility:
    """1 - SSR/SST from the low-level solver matches the model's reported R²."""

    def test_no_controls_fwl_path_r2_matches_raw_ssr_sst(self):
        factors, _, targets = _random_panel(n_factors=1, n_controls=0, nan_gap=True)
        result = RollingOLS(
            window=15, min_periods=10, precision="double", mode="batched"
        ).fit_transform(factors, targets)
        design = factors[["f0"]]
        raw = rolling_joint_solve(
            targets,
            design,
            window=15,
            min_periods=10,
            expanding=False,
            fit_intercept=True,
            warn_singular=False,
        )
        expected_r2 = 1.0 - raw.ssr / raw.sst
        actual_r2 = result.get_r2("f0").to_numpy()
        valid = np.isfinite(expected_r2) & np.isfinite(actual_r2)
        np.testing.assert_allclose(actual_r2[valid], expected_r2[valid], rtol=1e-9, atol=1e-12)

    def test_with_controls_joint_path_r2_matches_raw_ssr_sst(self):
        factors, controls, targets = _random_panel(n_factors=2, n_controls=2, nan_gap=True)
        result = RollingOLS(
            window=18, min_periods=12, mode="joint", precision="double"
        ).fit_transform(factors, targets, controls=controls)
        design = pd.concat([controls, factors], axis=1)
        raw = rolling_joint_solve(
            targets,
            design,
            window=18,
            min_periods=12,
            expanding=False,
            fit_intercept=True,
            warn_singular=False,
        )
        expected_r2 = 1.0 - raw.ssr / raw.sst
        for factor in factors.columns:
            actual_r2 = result.get_r2(factor).to_numpy()
            valid = np.isfinite(expected_r2) & np.isfinite(actual_r2)
            np.testing.assert_allclose(actual_r2[valid], expected_r2[valid], rtol=1e-9, atol=1e-12)


class TestScaleEquivariance:
    def test_scaling_factor_scales_its_beta_by_inverse_and_leaves_the_rest(self):
        factors, controls, targets = _random_panel(n_factors=1, n_controls=1, seed=11)
        scale = 4.0

        original = RollingOLS(window=15, precision="double", mode="batched").fit_transform(
            factors, targets, controls=controls, return_control_betas=True
        )
        scaled_factors = factors.copy()
        scaled_factors["f0"] *= scale
        scaled = RollingOLS(window=15, precision="double", mode="batched").fit_transform(
            scaled_factors, targets, controls=controls, return_control_betas=True
        )

        beta_original = original.get_beta("f0")
        beta_scaled = scaled.get_beta("f0")
        both_valid = beta_original.notna() & beta_scaled.notna()
        np.testing.assert_allclose(
            beta_scaled.to_numpy()[both_valid.to_numpy()],
            beta_original.to_numpy()[both_valid.to_numpy()] / scale,
            rtol=1e-8,
            atol=1e-10,
        )

        intercept_original = original.get_intercept("f0")
        intercept_scaled = scaled.get_intercept("f0")
        np.testing.assert_allclose(
            intercept_scaled.to_numpy()[both_valid.to_numpy()],
            intercept_original.to_numpy()[both_valid.to_numpy()],
            rtol=1e-8,
            atol=1e-10,
        )

        r2_original = original.get_r2("f0")
        r2_scaled = scaled.get_r2("f0")
        np.testing.assert_allclose(
            r2_scaled.to_numpy()[both_valid.to_numpy()],
            r2_original.to_numpy()[both_valid.to_numpy()],
            rtol=1e-8,
            atol=1e-10,
        )

        cb_original = original.get_control_beta("f0", "c0")
        cb_scaled = scaled.get_control_beta("f0", "c0")
        np.testing.assert_allclose(
            cb_scaled.to_numpy()[both_valid.to_numpy()],
            cb_original.to_numpy()[both_valid.to_numpy()],
            rtol=1e-8,
            atol=1e-10,
        )


class TestLocationInvariance:
    def test_shifting_factor_leaves_beta_unchanged_with_intercept(self):
        """A direct probe of the original intercept defect: with
        fit_intercept=True, adding a constant to a factor must not move its
        beta — the intercept absorbs the shift.
        """
        factors, controls, targets = _random_panel(n_factors=1, n_controls=1, seed=12)
        shift = 100.0

        original = RollingOLS(
            window=15, fit_intercept=True, precision="double", mode="batched"
        ).fit_transform(factors, targets, controls=controls)
        shifted_factors = factors.copy()
        shifted_factors["f0"] += shift
        shifted = RollingOLS(
            window=15, fit_intercept=True, precision="double", mode="batched"
        ).fit_transform(shifted_factors, targets, controls=controls)

        beta_original = original.get_beta("f0")
        beta_shifted = shifted.get_beta("f0")
        both_valid = beta_original.notna() & beta_shifted.notna()
        np.testing.assert_allclose(
            beta_shifted.to_numpy()[both_valid.to_numpy()],
            beta_original.to_numpy()[both_valid.to_numpy()],
            rtol=1e-8,
            atol=1e-8,
        )

        intercept_original = original.get_intercept("f0").to_numpy()
        intercept_shifted = shifted.get_intercept("f0").to_numpy()
        beta_values = beta_original.to_numpy()
        expected_intercept_shifted = intercept_original - beta_values * shift
        np.testing.assert_allclose(
            intercept_shifted[both_valid.to_numpy()],
            expected_intercept_shifted[both_valid.to_numpy()],
            rtol=1e-6,
            atol=1e-6,
        )


class TestPermutationInvariance:
    def test_reordering_targets_permutes_outputs_only(self):
        factors, controls, targets = _random_panel(n_targets=5, seed=13)
        order = ["a3", "a0", "a4", "a1", "a2"]

        original = RollingOLS(window=15, precision="double", mode="batched").fit_transform(
            factors, targets, controls=controls, return_control_betas=True
        )
        permuted = RollingOLS(window=15, precision="double", mode="batched").fit_transform(
            factors, targets[order], controls=controls, return_control_betas=True
        )

        for factor in factors.columns:
            for accessor in ("get_beta", "get_intercept", "get_residuals", "get_r2", "get_n_used"):
                expected = getattr(original, accessor)(factor)
                actual = getattr(permuted, accessor)(factor)[targets.columns]
                np.testing.assert_array_equal(actual.to_numpy(), expected.to_numpy())
            for control in controls.columns:
                expected = original.get_control_beta(factor, control)
                actual = permuted.get_control_beta(factor, control)[targets.columns]
                np.testing.assert_array_equal(actual.to_numpy(), expected.to_numpy())


class TestCrossAssetIsolation:
    def test_adding_all_nan_asset_leaves_other_assets_bitwise_unchanged(self):
        factors, controls, targets = _random_panel(n_targets=3, seed=14)

        baseline = RollingOLS(window=15, precision="double", mode="batched").fit_transform(
            factors, targets, controls=controls
        )
        augmented_targets = targets.copy()
        augmented_targets["a_nan"] = np.nan
        augmented = RollingOLS(window=15, precision="double", mode="batched").fit_transform(
            factors, augmented_targets, controls=controls
        )

        for factor in factors.columns:
            expected = baseline.get_beta(factor)
            actual = augmented.get_beta(factor)[targets.columns]
            np.testing.assert_array_equal(actual.to_numpy(), expected.to_numpy())
            assert augmented.get_beta(factor)["a_nan"].isna().all()


class TestChunkInvariance:
    @pytest.mark.parametrize("asset_chunk_size", [1, 7, 1000])
    def test_chunk_size_does_not_change_results(self, asset_chunk_size):
        factors, controls, targets = _random_panel(n_targets=9, seed=15)
        baseline = RollingOLS(
            window=15, precision="double", asset_chunk_size=100, mode="batched"
        ).fit_transform(factors, targets, controls=controls)
        chunked = RollingOLS(
            window=15, precision="double", asset_chunk_size=asset_chunk_size, mode="batched"
        ).fit_transform(factors, targets, controls=controls)

        for factor in factors.columns:
            np.testing.assert_array_equal(
                chunked.get_beta(factor).to_numpy(), baseline.get_beta(factor).to_numpy()
            )


class TestSubsetInvariance:
    def test_transform_subset_matches_transform_full_restricted(self):
        factors, controls, targets = _random_panel(n_targets=4, seed=16)
        model = RollingOLS(window=15, precision="double", mode="batched").fit(
            factors, controls=controls
        )

        full_result = model.transform(targets)
        subset_result = model.transform(targets[["a0", "a1"]])

        for factor in factors.columns:
            expected = full_result.get_beta(factor)[["a0", "a1"]]
            actual = subset_result.get_beta(factor)
            # Different target sets can be grouped into different
            # missingness patterns (task 12), which changes floating-point
            # summation order — allow float64 rounding, not a real deviation.
            np.testing.assert_allclose(
                actual.to_numpy(), expected.to_numpy(), rtol=1e-12, atol=1e-12, equal_nan=True
            )


class TestPathInvariance:
    """FWL and joint solves agree to 1e-10 (task 12)."""

    @pytest.mark.parametrize("fit_intercept", [True, False])
    def test_fwl_matches_joint_for_every_factor(self, fit_intercept):
        factors, controls, targets = _random_panel(n_factors=3, n_controls=2, nan_gap=True, seed=17)

        fwl = rolling_fwl_solve(
            targets,
            factors,
            controls,
            window=15,
            min_periods=10,
            expanding=False,
            fit_intercept=fit_intercept,
            warn_singular=False,
        )
        for position, factor in enumerate(factors.columns):
            design = pd.concat([controls, factors[[factor]]], axis=1)
            joint = rolling_joint_solve(
                targets,
                design,
                window=15,
                min_periods=10,
                expanding=False,
                fit_intercept=fit_intercept,
                warn_singular=False,
            )
            np.testing.assert_allclose(
                fwl.factor_coef[:, position], joint.coef[:, -1], rtol=0, atol=1e-10
            )


class TestCadenceInvariance:
    @pytest.mark.parametrize("estimate_every", [1, 3, 7])
    def test_cadence_matches_full_fit_sliced(self, estimate_every):
        factors, controls, targets = _random_panel(n_observations=80, seed=18)

        full = RollingOLS(
            window=15, precision="double", estimate_every=1, mode="batched"
        ).fit_transform(factors, targets, controls=controls)
        cadenced = RollingOLS(
            window=15, precision="double", estimate_every=estimate_every, mode="batched"
        ).fit_transform(factors, targets, controls=controls)

        for factor in factors.columns:
            beta_cadenced = cadenced.get_beta(factor)
            kept = beta_cadenced.notna().to_numpy()
            beta_full = full.get_beta(factor).to_numpy()
            np.testing.assert_array_equal(beta_cadenced.to_numpy()[kept], beta_full[kept])


class TestLazyInvariance:
    def test_accessor_values_independent_of_call_order(self):
        factors, controls, targets = _random_panel(n_factors=2, n_controls=1, seed=19)

        forward = RollingOLS(
            window=15, hac_lags=3, precision="double", cache_size=1, mode="batched"
        ).fit_transform(factors, targets, controls=controls, return_control_betas=True)
        beta_f0_first = forward.get_beta("f0")
        se_f0_first = forward.get_se("f0")
        r2_f1_first = forward.get_r2("f1")

        backward = RollingOLS(
            window=15, hac_lags=3, precision="double", cache_size=1, mode="batched"
        ).fit_transform(factors, targets, controls=controls, return_control_betas=True)
        r2_f1_second = backward.get_r2("f1")
        se_f0_second = backward.get_se("f0")
        beta_f0_second = backward.get_beta("f0")

        np.testing.assert_array_equal(beta_f0_first.to_numpy(), beta_f0_second.to_numpy())
        np.testing.assert_array_equal(se_f0_first.to_numpy(), se_f0_second.to_numpy())
        np.testing.assert_array_equal(r2_f1_first.to_numpy(), r2_f1_second.to_numpy())

        # Re-requesting an evicted factor (cache_size=1) must reproduce the
        # exact same value as the first computation, not something derived
        # from stale state.
        se_f0_third = forward.get_se("f0")
        np.testing.assert_array_equal(se_f0_first.to_numpy(), se_f0_third.to_numpy())


class TestDtypePrecision:
    def test_mixed_matches_double_within_tolerance(self):
        factors, controls, targets = _random_panel(n_factors=2, n_controls=1, seed=20)

        result_mixed = RollingOLS(window=15, precision="mixed", mode="batched").fit_transform(
            factors, targets, controls=controls
        )
        result_double = RollingOLS(window=15, precision="double", mode="batched").fit_transform(
            factors, targets, controls=controls
        )

        for factor in factors.columns:
            beta_mixed = result_mixed.get_beta(factor)
            beta_double = result_double.get_beta(factor)
            both_valid = beta_mixed.notna() & beta_double.notna()
            np.testing.assert_allclose(
                beta_mixed.to_numpy()[both_valid.to_numpy()],
                beta_double.to_numpy()[both_valid.to_numpy()],
                rtol=0,
                atol=1e-4,
            )

    def test_mixed_output_dtype_is_float64(self):
        """Mixed precision computes in float64; output frames are float64."""
        factors, controls, targets = _random_panel(n_factors=1, n_controls=1, seed=21)
        result_mixed = RollingOLS(window=15, precision="mixed", mode="batched").fit_transform(
            factors, targets, controls=controls
        )
        assert result_mixed.get_beta("f0").to_numpy().dtype == np.float64
        assert result_mixed.get_r2("f0").to_numpy().dtype == np.float64


@pytest.mark.parametrize("mode", ["batched", "joint"])
@pytest.mark.parametrize("lambda_", [0.0, 1e-4])
@pytest.mark.parametrize("controls", [True, False])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_fitted_values_residual_coherence(
    mode: str, lambda_: float, controls: bool, fit_intercept: bool
):
    """get_fitted_values() + get_residuals() == y on the complete-case sample."""
    factors, ctrls, targets = _random_panel(n_observations=100, n_factors=2, n_controls=1)
    if not controls:
        ctrls = None

    ols = RollingOLS(
        window=20,
        lambda_=lambda_,
        mode=mode,
        fit_intercept=fit_intercept,
    )
    result = ols.fit(factors, controls=ctrls).transform(targets)

    for factor in factors.columns:
        if mode == "joint":
            # Just pass the factor, warning is emitted if we pass factor in joint mode,
            # but wait, the accessor accepts it with a warning. We can pass None in joint mode.
            pass

        # Get fitted values and residuals
        fitted = result.get_fitted_values(factor if mode == "batched" else None)
        residuals = result.get_residuals(factor)

        # Test invariant
        coherence = fitted + residuals

        # We only expect equality on complete-case sample (non-NaNs)
        mask = ~fitted.isna()

        np.testing.assert_allclose(
            coherence[mask].values,
            targets[mask].values,
            atol=1e-12,
        )


def test_fitted_values_lag_independence():
    """Fitted values do not inherit lag_signal; signals do."""
    factors, ctrls, targets = _random_panel(n_observations=50, n_factors=1, n_controls=0)

    ols_contemp = RollingOLS(window=10, lag_signal=False, mode="joint")
    res_contemp = ols_contemp.fit(factors).transform(targets)

    ols_lagged = RollingOLS(window=10, lag_signal=True, mode="joint")
    res_lagged = ols_lagged.fit(factors).transform(targets)

    # Fitted values should be identical
    np.testing.assert_allclose(
        res_contemp.get_fitted_values().values,
        res_lagged.get_fitted_values().values,
        equal_nan=True,
    )

    # Signals should differ (except maybe at NaNs)
    sig_contemp = res_contemp.get_signal(factors.columns[0]).fillna(0)
    sig_lagged = res_lagged.get_signal(factors.columns[0]).fillna(0)
    assert not np.allclose(sig_contemp.values, sig_lagged.values)


def test_fitted_values_batched_vs_joint():
    """Batched requires factor and varies; Joint ignores factor and doesn't vary."""
    factors, ctrls, targets = _random_panel(n_observations=50, n_factors=2, n_controls=1)

    # Batched
    res_batched = (
        RollingOLS(window=10, mode="batched").fit(factors, controls=ctrls).transform(targets)
    )

    # Must raise if no factor specified and >1 factors exist
    with pytest.raises(ValueError, match="requires a `factor` argument"):
        res_batched.get_fitted_values()

    fit_f1 = res_batched.get_fitted_values(factors.columns[0])
    fit_f2 = res_batched.get_fitted_values(factors.columns[1])

    # They should differ because each includes a different factor
    assert not np.allclose(fit_f1.fillna(0).values, fit_f2.fillna(0).values)

    # Joint
    res_joint = RollingOLS(window=10, mode="joint").fit(factors, controls=ctrls).transform(targets)

    # factor ignored in joint, they should all be identical
    with pytest.warns(DeprecationWarning, match="ignored in joint mode"):
        fit_joint_f1 = res_joint.get_fitted_values(factors.columns[0])

    fit_joint_none = res_joint.get_fitted_values()

    np.testing.assert_allclose(
        fit_joint_f1.values,
        fit_joint_none.values,
        equal_nan=True,
    )


def test_fitted_values_degenerate_single_factor():
    """get_fitted_values() equals get_signal() when no controls and fit_intercept=False."""
    factors, ctrls, targets = _random_panel(n_observations=50, n_factors=1, n_controls=0)

    res = (
        RollingOLS(window=10, mode="joint", fit_intercept=False, lag_signal=False)
        .fit(factors)
        .transform(targets)
    )

    f1 = factors.columns[0]
    np.testing.assert_allclose(
        res.get_fitted_values().values,
        res.get_signal(f1).values,
        equal_nan=True,
    )
