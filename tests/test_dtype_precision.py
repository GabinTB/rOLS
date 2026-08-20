"""Tests for the precision policy (issue #25).

Covers all three precision modes — double, mixed, single — and verifies:
1.  Attribute correctness (precision, storage_dtype, compute_dtype)
2.  Validation (invalid precision values)
3.  Storage dtype propagation to retained arrays
4.  Compute dtype propagation through solvers
5.  Result attributes round-trip
6.  Double vs mixed numerical agreement (same compute_dtype)
7.  Double vs single numerical agreement (within tolerance)
8.  cond_warn_threshold auto-selection
9.  estimate_memory respects storage_dtype
10. Output dtype matches compute_dtype
11. HAC SE dtype consistency
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rols import RollingOLS


def _panel(
    n_observations: int = 60,
    n_factors: int = 1,
    n_controls: int = 0,
    n_targets: int = 3,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
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


# ---------------------------------------------------------------------------
# 1. Attribute correctness
# ---------------------------------------------------------------------------


class TestAttributeCorrectness:
    def test_double_attributes(self):
        m = RollingOLS(window=20, precision="double")
        assert m.precision == "double"
        assert m.storage_dtype == np.float64
        assert m.compute_dtype == np.float64

    def test_mixed_attributes(self):
        m = RollingOLS(window=20, precision="mixed")
        assert m.precision == "mixed"
        assert m.storage_dtype == np.float32
        assert m.compute_dtype == np.float64

    def test_single_attributes(self):
        m = RollingOLS(window=20, precision="single")
        assert m.precision == "single"
        assert m.storage_dtype == np.float32
        assert m.compute_dtype == np.float32


# ---------------------------------------------------------------------------
# 2. Validation
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize("bad", ["float32", "float64", "half", "", 0, None, True])
    def test_invalid_precision_raises(self, bad):
        with pytest.raises(ValueError, match="precision must be"):
            RollingOLS(window=20, precision=bad)


# ---------------------------------------------------------------------------
# 3. Storage dtype propagation
# ---------------------------------------------------------------------------


class TestStorageDtypePropagation:
    @pytest.mark.parametrize(
        "precision,expected",
        [("double", np.float64), ("mixed", np.float32), ("single", np.float32)],
    )
    def test_factors_stored_in_storage_dtype(self, precision, expected):
        factors, _, targets = _panel()
        model = RollingOLS(window=15, precision=precision, mode="batched")
        model.fit(factors)
        # The model stores factors in self._factors with storage_dtype
        stored = model._factors
        assert stored is not None
        assert stored.to_numpy().dtype == expected

    @pytest.mark.parametrize(
        "precision,expected",
        [("double", np.float64), ("mixed", np.float32), ("single", np.float32)],
    )
    def test_controls_stored_in_storage_dtype(self, precision, expected):
        factors, controls, targets = _panel(n_controls=1)
        model = RollingOLS(window=15, precision=precision, mode="batched")
        model.fit(factors, controls=controls)
        stored = model._controls_fitted
        assert stored is not None
        assert stored.to_numpy().dtype == expected


# ---------------------------------------------------------------------------
# 4. Compute dtype propagation
# ---------------------------------------------------------------------------


class TestComputeDtypePropagation:
    @pytest.mark.parametrize("precision", ["double", "mixed", "single"])
    def test_beta_dtype_matches_compute_dtype(self, precision):
        factors, controls, targets = _panel(n_controls=1)
        model = RollingOLS(window=15, precision=precision, mode="batched")
        result = model.fit_transform(factors, targets, controls=controls)
        beta_arr = result.get_beta("f0").to_numpy()
        assert beta_arr.dtype == model.compute_dtype

    @pytest.mark.parametrize("precision", ["double", "mixed", "single"])
    def test_signal_dtype_matches_compute_dtype(self, precision):
        factors, _, targets = _panel()
        model = RollingOLS(window=15, precision=precision, mode="batched")
        result = model.fit_transform(factors, targets)
        signal_arr = result.get_signal("f0").to_numpy()
        assert signal_arr.dtype == model.compute_dtype


# ---------------------------------------------------------------------------
# 5. Result attributes round-trip
# ---------------------------------------------------------------------------


class TestResultAttributes:
    @pytest.mark.parametrize("precision", ["double", "mixed", "single"])
    def test_result_carries_precision_fields(self, precision):
        factors, _, targets = _panel()
        model = RollingOLS(window=15, precision=precision, mode="batched")
        result = model.fit_transform(factors, targets)
        assert result.precision == precision
        assert result.storage_dtype == np.dtype(model.storage_dtype)
        assert result.compute_dtype == np.dtype(model.compute_dtype)


# ---------------------------------------------------------------------------
# 6. Double vs mixed numerical agreement
# ---------------------------------------------------------------------------


class TestDoubleVsMixed:
    """Mixed uses float64 compute, so results must match double exactly
    (up to float32 → float64 input conversion noise, which is zero for
    synthetically generated data that fits in float32)."""

    def test_beta_matches_within_input_conversion_noise(self):
        """Mixed stores inputs in float32 then computes in float64; the
        float32 → float64 input conversion introduces ~1e-7 noise."""
        factors, controls, targets = _panel(n_controls=1, seed=10)
        r_double = RollingOLS(window=15, precision="double", mode="batched").fit_transform(
            factors, targets, controls=controls
        )
        r_mixed = RollingOLS(window=15, precision="mixed", mode="batched").fit_transform(
            factors, targets, controls=controls
        )
        for factor in factors.columns:
            b_d = r_double.get_beta(factor)
            b_m = r_mixed.get_beta(factor)
            both = b_d.notna() & b_m.notna()
            np.testing.assert_allclose(
                b_m.to_numpy()[both.to_numpy()],
                b_d.to_numpy()[both.to_numpy()],
                rtol=1e-5,
                atol=1e-6,
            )

    def test_r2_matches_within_tolerance(self):
        factors, _, targets = _panel(seed=11)
        r_double = RollingOLS(window=15, precision="double", mode="batched").fit_transform(
            factors, targets
        )
        r_mixed = RollingOLS(window=15, precision="mixed", mode="batched").fit_transform(
            factors, targets
        )
        for factor in factors.columns:
            r2_d = r_double.get_r2(factor)
            r2_m = r_mixed.get_r2(factor)
            both = r2_d.notna() & r2_m.notna()
            np.testing.assert_allclose(
                r2_m.to_numpy()[both.to_numpy()],
                r2_d.to_numpy()[both.to_numpy()],
                rtol=1e-5,
                atol=1e-6,
            )


# ---------------------------------------------------------------------------
# 7. Double vs single numerical agreement (within tolerance)
# ---------------------------------------------------------------------------


class TestDoubleVsSingle:
    """Single uses float32 compute, so results differ by ~1e-4 to 1e-3."""

    def test_beta_within_float32_tolerance(self):
        factors, controls, targets = _panel(n_controls=1, seed=20)
        r_double = RollingOLS(window=15, precision="double", mode="batched").fit_transform(
            factors, targets, controls=controls
        )
        r_single = RollingOLS(window=15, precision="single", mode="batched").fit_transform(
            factors, targets, controls=controls
        )
        for factor in factors.columns:
            b_d = r_double.get_beta(factor)
            b_s = r_single.get_beta(factor)
            both = b_d.notna() & b_s.notna()
            np.testing.assert_allclose(
                b_s.to_numpy()[both.to_numpy()],
                b_d.to_numpy()[both.to_numpy()],
                rtol=0,
                atol=1e-3,
            )

    def test_single_output_dtype_is_float32(self):
        factors, _, targets = _panel(seed=21)
        result = RollingOLS(window=15, precision="single", mode="batched").fit_transform(
            factors, targets
        )
        assert result.get_beta("f0").to_numpy().dtype == np.float32
        assert result.get_r2("f0").to_numpy().dtype == np.float32


# ---------------------------------------------------------------------------
# 8. cond_warn_threshold auto-selection
# ---------------------------------------------------------------------------


class TestCondWarnThreshold:
    def test_double_default_threshold(self):
        m = RollingOLS(window=20, precision="double")
        assert m.cond_warn_threshold == 1e10

    def test_mixed_default_threshold(self):
        m = RollingOLS(window=20, precision="mixed")
        assert m.cond_warn_threshold == 1e10  # compute is float64

    def test_single_default_threshold(self):
        m = RollingOLS(window=20, precision="single")
        assert m.cond_warn_threshold == 1e5  # compute is float32

    def test_explicit_override(self):
        m = RollingOLS(window=20, precision="single", cond_warn_threshold=42.0)
        assert m.cond_warn_threshold == 42.0


# ---------------------------------------------------------------------------
# 9. estimate_memory respects storage_dtype
# ---------------------------------------------------------------------------


class TestEstimateMemory:
    def test_mixed_uses_less_memory_than_double(self):
        factors, _, targets = _panel(n_observations=100, n_targets=50)
        m_double = RollingOLS(window=20, precision="double")
        m_mixed = RollingOLS(window=20, precision="mixed")
        mem_double = m_double.estimate_memory(targets, factors)
        mem_mixed = m_mixed.estimate_memory(targets, factors)
        # float32 storage uses half the bytes of float64 storage
        assert mem_mixed["total"] < mem_double["total"]

    def test_single_same_as_mixed_storage(self):
        factors, _, targets = _panel(n_observations=100, n_targets=50)
        m_mixed = RollingOLS(window=20, precision="mixed")
        m_single = RollingOLS(window=20, precision="single")
        mem_mixed = m_mixed.estimate_memory(targets, factors)
        mem_single = m_single.estimate_memory(targets, factors)
        # Both use float32 storage, so estimates should be equal
        assert mem_mixed["total"] == mem_single["total"]


# ---------------------------------------------------------------------------
# 10. Output dtype matches compute_dtype
# ---------------------------------------------------------------------------


class TestOutputDtype:
    @pytest.mark.parametrize("precision", ["double", "mixed", "single"])
    def test_intercept_dtype(self, precision):
        factors, _, targets = _panel()
        model = RollingOLS(window=15, precision=precision, mode="batched")
        result = model.fit_transform(factors, targets)
        arr = result.get_intercept("f0").to_numpy()
        assert arr.dtype == model.compute_dtype

    @pytest.mark.parametrize("precision", ["double", "mixed", "single"])
    def test_r2_dtype(self, precision):
        factors, _, targets = _panel()
        model = RollingOLS(window=15, precision=precision, mode="batched")
        result = model.fit_transform(factors, targets)
        arr = result.get_r2("f0").to_numpy()
        assert arr.dtype == model.compute_dtype

    @pytest.mark.parametrize("precision", ["double", "mixed", "single"])
    def test_partial_r2_dtype(self, precision):
        factors, controls, targets = _panel(n_controls=1)
        model = RollingOLS(window=15, precision=precision, mode="batched")
        result = model.fit_transform(factors, targets, controls=controls)
        arr = result.get_partial_r2("f0").to_numpy()
        assert arr.dtype == model.compute_dtype


# ---------------------------------------------------------------------------
# 11. HAC SE dtype consistency
# ---------------------------------------------------------------------------


class TestHACDtype:
    @pytest.mark.parametrize("precision", ["double", "mixed", "single"])
    def test_se_dtype_matches_compute_dtype(self, precision):
        factors, _, targets = _panel()
        model = RollingOLS(window=15, hac_lags=2, precision=precision, mode="batched")
        result = model.fit_transform(factors, targets)
        se_arr = result.get_se("f0").to_numpy()
        assert se_arr.dtype == model.compute_dtype

    @pytest.mark.parametrize("precision", ["double", "mixed", "single"])
    def test_tstat_dtype_matches_compute_dtype(self, precision):
        factors, _, targets = _panel()
        model = RollingOLS(window=15, hac_lags=2, precision=precision, mode="batched")
        result = model.fit_transform(factors, targets)
        tstat_arr = result.get_tstat("f0").to_numpy()
        assert tstat_arr.dtype == model.compute_dtype

    def test_hac_se_single_vs_double_within_tolerance(self):
        factors, _, targets = _panel(seed=30)
        r_double = RollingOLS(
            window=15, hac_lags=2, precision="double", mode="batched"
        ).fit_transform(factors, targets)
        r_single = RollingOLS(
            window=15, hac_lags=2, precision="single", mode="batched"
        ).fit_transform(factors, targets)
        se_d = r_double.get_se("f0")
        se_s = r_single.get_se("f0")
        both = se_d.notna() & se_s.notna()
        np.testing.assert_allclose(
            se_s.to_numpy()[both.to_numpy()],
            se_d.to_numpy()[both.to_numpy()],
            rtol=0,
            atol=1e-2,
        )
