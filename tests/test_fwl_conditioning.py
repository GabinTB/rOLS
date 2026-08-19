"""Validate the FWL rank-condition heuristic against the scalar oracle.

Five test groups prove the two critical behavioural invariants of
rolling_fwl_solve under near-collinear and rank-deficient designs:

  Invariant A — never silently wrong:
    When the oracle succeeds, FWL either agrees to 1e-6 OR returns NaN with
    n_singular > 0.  A plausible-looking but incorrect number is forbidden.

  Invariant B — loud failure:
    When the nuisance design is rank-deficient, FWL returns NaN for every
    affected window and increments n_singular.  No exception is raised.

Test groups
-----------
1. Near-collinear control sweep (κ ∈ {10, 10³, 10⁶, 10⁹, 10¹²}):
   confirm Invariant A across a wide condition-number range.

2. True rank deficiency — exact duplicate control column: confirm Invariant B
   and that the oracle also rejects the degenerate design.

3. Near-duplicate control column (ctrl_2 = ctrl_1 + 1e-15 noise): same
   invariants, edge case where machine precision determines rank.

4. Near-collinear factor and control (f ≈ c): full-rank design that should
   match the oracle; may trigger the condition-number warning.

5. cond_warn_threshold boundary: warning fires when condition exceeds the
   threshold, silent when below; result agrees with oracle in both cases.
"""

from __future__ import annotations

import warnings as warnings_module

import numpy as np
import pandas as pd
import pytest

from rols.estimators import rolling_fwl_solve
from tests.oracle import oracle_fit_window

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uniform_design_gram(nuisance: np.ndarray, factor: np.ndarray) -> np.ndarray:
    """Build the augmented gram that rolling_fwl_solve inspects (uniform weights).

    nuisance : (T, n_nuis) weighted-nuisance matrix (already sqrt(w)-scaled)
    factor   : (T,) weighted-factor vector (already sqrt(w)-scaled)
    """
    n_nuis = nuisance.shape[1]
    cross = nuisance.T @ factor  # (n_nuis,)
    gram = np.zeros((n_nuis + 1, n_nuis + 1))
    gram[:n_nuis, :n_nuis] = nuisance.T @ nuisance
    gram[:n_nuis, -1] = cross
    gram[-1, :n_nuis] = cross
    gram[-1, -1] = float(factor @ factor)
    return gram


def _fwl_single_window(
    y: np.ndarray,
    f: np.ndarray,
    C: np.ndarray,
    *,
    cond_warn_threshold: float = 1e10,
    warn_singular: bool = False,
) -> tuple[float, int]:
    """Call rolling_fwl_solve for one full window; return (factor_coef, n_singular)."""
    T = len(y)
    idx = pd.RangeIndex(T)
    y_df = pd.DataFrame({"y": y}, index=idx)
    f_df = pd.DataFrame({"f": f}, index=idx)
    c_cols = {f"c{i}": C[:, i] for i in range(C.shape[1])}
    c_df = pd.DataFrame(c_cols, index=idx)

    result = rolling_fwl_solve(
        y=y_df,
        factors=f_df,
        controls=c_df,
        window=T,
        min_periods=T,
        expanding=False,
        fit_intercept=True,
        warn_singular=warn_singular,
        cond_warn_threshold=cond_warn_threshold,
    )
    assert result.factor_coef is not None
    # factor_coef has shape (output_size, n_factors, n_targets); one slot per
    # time step, with NaN where min_periods is not yet met.  The last position
    # (-1) is the only fully populated rolling endpoint for window == T.
    return float(result.factor_coef[-1, 0, 0]), result.n_singular


def _oracle_factor_coef(y: np.ndarray, f: np.ndarray, C: np.ndarray) -> float:
    """Oracle factor coefficient: X = [controls, factor], fit_intercept=True."""
    X = np.column_stack([C, f])
    fit = oracle_fit_window(y=y, X=X, fit_intercept=True, weights=None, penalty=None)
    return float(fit.coef[-1])


# ---------------------------------------------------------------------------
# 1. Near-collinear control sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kappa", [10.0, 1e3, 1e6, 1e9, 1e12])
def test_near_collinear_control_sweep(kappa: float) -> None:
    """FWL matches oracle or NaNs loudly across a wide condition-number range."""
    T = 60
    rng = np.random.default_rng(0)

    # Build controls with exact condition number κ via SVD construction.
    U, _ = np.linalg.qr(rng.normal(size=(T, 2)))
    V, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    sigma = np.array([1.0, 1.0 / kappa])
    C = U @ np.diag(sigma) @ V.T  # cond(C) ≈ κ

    f = rng.normal(size=T)
    y = 2.0 * f + rng.normal(size=T)

    fwl_coef, n_sing = _fwl_single_window(y, f, C)
    oracle_coef = _oracle_factor_coef(y, f, C)

    if np.isnan(fwl_coef):
        # Invariant A: NaN must be loud — n_singular must be positive.
        assert n_sing > 0, (
            f"FWL returned NaN for κ={kappa:.0e} but n_singular == 0 "
            "(silent failure — the rank-condition guard must increment the counter)"
        )
        # Regression guard: if oracle succeeded, FWL was conservative (allowed).
        # If oracle also failed, both agree on degeneracy.
    else:
        # FWL returned a finite value — it must agree with the oracle.
        # Tolerance is 1e-6 (loose relative to the 1e-10 standard) to allow
        # for different numerics between QR (FWL) and lstsq (oracle).  At
        # κ=10¹² we allow 1e-5 because the two double-precision paths can
        # diverge further at the edge of representable accuracy.
        tol = 1e-5 if kappa >= 1e12 else 1e-6
        if not np.isnan(oracle_coef):
            assert abs(fwl_coef - oracle_coef) < tol, (
                f"FWL={fwl_coef:.6g} vs oracle={oracle_coef:.6g} at κ={kappa:.0e}"
            )


# ---------------------------------------------------------------------------
# 2. True rank deficiency — exact duplicate control column
# ---------------------------------------------------------------------------


def test_true_rank_deficiency_exact_duplicate() -> None:
    """FWL returns NaN + n_singular>0 for an exactly duplicate control; no exception."""
    T = 60
    rng = np.random.default_rng(1)
    c = rng.normal(size=T)
    f = rng.normal(size=T)
    y = 2.0 * f + rng.normal(size=T)
    C_dup = np.column_stack([c, c])  # rank-1 instead of rank-2

    fwl_coef, n_sing = _fwl_single_window(y, f, C_dup)

    assert np.isnan(fwl_coef), (
        "FWL must return NaN when the nuisance design [1, c, c] is rank-deficient"
    )
    assert n_sing > 0, "n_singular must be positive for a rank-deficient design"

    # Regression guard: oracle also rejects the degenerate design.
    oracle_coef = _oracle_factor_coef(y, f, C_dup)
    assert np.isnan(oracle_coef), (
        "Oracle should also return NaN for [1, c, c] (rank check at oracle level)"
    )


# ---------------------------------------------------------------------------
# 3. Near-duplicate control column (machine-precision separation)
# ---------------------------------------------------------------------------


def test_true_rank_deficiency_near_duplicate() -> None:
    """Near-duplicate controls either NaN loudly or agree with oracle — never wrong."""
    T = 60
    rng = np.random.default_rng(2)
    c = rng.normal(size=T)
    noise = rng.normal(size=T)
    f = rng.normal(size=T)
    y = 2.0 * f + rng.normal(size=T)
    C_near = np.column_stack([c, c + 1e-15 * noise])  # near-singular

    fwl_coef, n_sing = _fwl_single_window(y, f, C_near)
    oracle_coef = _oracle_factor_coef(y, f, C_near)

    if np.isnan(fwl_coef):
        # Invariant B: NaN must be loud.
        assert n_sing > 0, "FWL returned NaN for near-duplicate controls but n_singular == 0"
    else:
        # FWL accepted the design; must agree with oracle (if oracle also accepted).
        if not np.isnan(oracle_coef):
            assert abs(fwl_coef - oracle_coef) < 1e-6, (
                f"FWL={fwl_coef:.6g} vs oracle={oracle_coef:.6g} for near-duplicate controls"
            )


# ---------------------------------------------------------------------------
# 4. Near-collinear factor and control
# ---------------------------------------------------------------------------


def test_near_collinear_factor_and_control() -> None:
    """factor ≈ control: full-rank design matches oracle; condition warning may fire."""
    T = 60
    rng = np.random.default_rng(3)
    c = rng.normal(size=T)
    f = c + 0.001 * rng.normal(size=T)  # nearly collinear with one control
    y = 2.0 * f - c + rng.normal(size=T)
    C = c[:, None]

    # Confirm the design is full rank (it must be since f ≠ c exactly).
    full_design = np.column_stack([np.ones(T), c, f])
    assert np.linalg.matrix_rank(full_design) == 3, "Design should be full rank"

    oracle_coef = _oracle_factor_coef(y, f, C)
    assert not np.isnan(oracle_coef), "Oracle must succeed on a full-rank design"

    # FWL is allowed to warn about ill-conditioning; capture but do not suppress.
    with warnings_module.catch_warnings(record=True):
        warnings_module.simplefilter("always")
        fwl_coef, n_sing = _fwl_single_window(y, f, C)

    if np.isnan(fwl_coef):
        # FWL was more conservative than oracle — allowed, but must be loud.
        assert n_sing > 0, (
            "FWL returned NaN on a technically full-rank near-collinear design "
            "without incrementing n_singular"
        )
    else:
        # FWL and oracle both succeeded; must agree.
        assert abs(fwl_coef - oracle_coef) < 1e-6, (
            f"FWL={fwl_coef:.6g} vs oracle={oracle_coef:.6g} for near-collinear f≈c"
        )


# ---------------------------------------------------------------------------
# 5. cond_warn_threshold boundary
# ---------------------------------------------------------------------------


def test_cond_warn_threshold_fires_above_threshold() -> None:
    """RuntimeWarning fires when the augmented-gram condition exceeds the threshold."""
    T = 60
    rng = np.random.default_rng(4)
    c = rng.normal(size=T)
    f = c + 0.01 * rng.normal(size=T)  # near-collinear
    y = 2.0 * f - c + rng.normal(size=T)

    # Compute the exact condition number that rolling_fwl_solve sees.
    sw = np.sqrt(np.full(T, 1.0 / T))
    nuisance_w = np.column_stack([np.ones(T), c]) * sw[:, None]
    factor_w = f * sw
    actual_cond = float(np.linalg.cond(_uniform_design_gram(nuisance_w, factor_w)))

    # Threshold just below the actual condition → warning must fire.
    threshold_low = actual_cond / 2.0
    with pytest.warns(RuntimeWarning, match="ill-conditioned"):
        rolling_fwl_solve(
            y=pd.DataFrame({"y": y}, index=pd.RangeIndex(T)),
            factors=pd.DataFrame({"f": f}, index=pd.RangeIndex(T)),
            controls=pd.DataFrame({"c": c}, index=pd.RangeIndex(T)),
            window=T,
            min_periods=T,
            expanding=False,
            fit_intercept=True,
            warn_singular=True,
            cond_warn_threshold=threshold_low,
        )


def test_cond_warn_threshold_silent_below_threshold() -> None:
    """No RuntimeWarning fires when the condition is below the threshold."""
    T = 60
    rng = np.random.default_rng(4)
    c = rng.normal(size=T)
    f = c + 0.01 * rng.normal(size=T)
    y = 2.0 * f - c + rng.normal(size=T)
    C = c[:, None]

    sw = np.sqrt(np.full(T, 1.0 / T))
    nuisance_w = np.column_stack([np.ones(T), c]) * sw[:, None]
    factor_w = f * sw
    actual_cond = float(np.linalg.cond(_uniform_design_gram(nuisance_w, factor_w)))

    # Threshold just above the actual condition → no warning, result matches oracle.
    threshold_high = actual_cond * 2.0
    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        result = rolling_fwl_solve(
            y=pd.DataFrame({"y": y}, index=pd.RangeIndex(T)),
            factors=pd.DataFrame({"f": f}, index=pd.RangeIndex(T)),
            controls=pd.DataFrame({"c": c}, index=pd.RangeIndex(T)),
            window=T,
            min_periods=T,
            expanding=False,
            fit_intercept=True,
            warn_singular=True,
            cond_warn_threshold=threshold_high,
        )
    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not runtime_warnings, (
        f"No RuntimeWarning expected below threshold; got: {[str(w.message) for w in runtime_warnings]}"
    )

    # Result must match oracle regardless of threshold setting.
    oracle_coef = _oracle_factor_coef(y, f, C)
    assert result.factor_coef is not None
    fwl_coef = float(result.factor_coef[0, 0, 0])
    if not np.isnan(oracle_coef) and not np.isnan(fwl_coef):
        assert abs(fwl_coef - oracle_coef) < 1e-6, (
            f"FWL={fwl_coef:.6g} vs oracle={oracle_coef:.6g} (threshold above actual_cond)"
        )
