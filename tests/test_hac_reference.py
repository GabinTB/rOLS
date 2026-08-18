"""Independent reference tests for the Ridge HAC sandwich.

Every test in this module derives the expected value from a path that is
structurally different from the production code:

  1. augmented-design lstsq reference (test_augmented_design_reference_*) —
     uses the stacked [√W Z ; √P] / [√W y ; 0] formulation and a full
     sandwich loop, never calling _factor_hac_standard_errors directly.

  2. factor-rescaling equivariance (test_rescaling_equivariance) — multiplying
     a factor by c must scale beta by 1/c and SE by 1/c while leaving the
     t-statistic invariant.  Failures point to a wrong exponent or missing
     unit conversion in the scales[-1]**2 step.

  3. lstsq Ridge coefficient comparison (test_ridge_coefficients_via_lstsq) —
     Ridge betas agree with np.linalg.lstsq on the augmented design without
     going through the production QR path.

  4. statsmodels WLS comparison (test_statsmodels_wls_hac) — unpenalized path
     with non-uniform weights, validated against statsmodels.WLS(cov_type="HAC").

  5. independently coded Newey-West loop (test_newey_west_reference_loop) —
     a deliberately slow Bartlett summation written directly from Newey & West
     (1987), used to cross-check the vectorized production implementation on
     a single window.

None of these tests import from the production sandwich implementation
(_factor_hac_standard_errors).  They compare only at the public API level
(get_se / rolling_hac_se) or at coefficients.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rols import RollingOLS
from rols.estimators import rolling_hac_se

# ──────────────────────────────────────────────────────────────────────────────
# Helper: build a small but realistic panel
# ──────────────────────────────────────────────────────────────────────────────


def _panel(
    T: int = 60,
    n_targets: int = 2,
    n_controls: int = 1,
    seed: int = 0,
    ewma_halflife: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, np.ndarray | None]:
    """Return (targets, factors, controls, weights)."""
    rng = np.random.default_rng(seed)
    idx = pd.RangeIndex(T)
    factors = pd.DataFrame({"f": rng.normal(size=T)}, index=idx)
    targets = pd.DataFrame(
        rng.normal(size=(T, n_targets)),
        index=idx,
        columns=[f"a{i}" for i in range(n_targets)],
    )
    controls: pd.DataFrame | None = None
    if n_controls > 0:
        controls = pd.DataFrame(
            rng.normal(size=(T, n_controls)),
            index=idx,
            columns=[f"c{i}" for i in range(n_controls)],
        )
    weights: np.ndarray | None = None
    if ewma_halflife is not None:
        # Mirror the weight scheme used by RollingOLS internally.
        lam = 0.5 ** (1.0 / ewma_halflife)
        raw = lam ** np.arange(T - 1, -1, -1, dtype=np.float64)
        weights = raw / raw.sum()
    return targets, factors, controls, weights


# ──────────────────────────────────────────────────────────────────────────────
# Reference sandwich: augmented-design construction
# ──────────────────────────────────────────────────────────────────────────────


def _reference_ridge_hac_se(
    y: np.ndarray,
    Z: np.ndarray,
    w: np.ndarray,
    P: np.ndarray,
    n_lags: int,
    fit_intercept: bool = True,
) -> float:
    """Ridge HAC SE for the last slope, derived from first principles.

    Uses the augmented-design lstsq path (structurally different from the
    production QR path) and a slow explicit Bartlett loop (structurally
    different from the vectorized production loop).

    Parameters
    ----------
    y : (n,) target observations (complete-case, not full-window)
    Z : (n, p) design matrix in original coordinates (intercept already present)
    w : (n,) normalized weights (sum to 1)
    P : (p, p) penalty matrix (diagonal; zero for unpenalized columns)
    n_lags : Bartlett lag count
    fit_intercept : whether the first column of Z is a constant intercept.
        When True, penalised columns are centred (mean subtracted) before
        scaling, mirroring the production standardisation logic.  When False,
        only scaling is applied — exactly as production code does (line 328).
    """
    n, p = Z.shape

    # ── 1. Standardize penalized columns ──────────────────────────────────────
    # Production centres only when fit_intercept=True (estimators.py:328-330).
    penalized = np.flatnonzero(np.diag(P) > 0)
    Zs = Z.copy()
    scales = np.ones(p)
    for col in penalized:
        if fit_intercept:
            Zs[:, col] -= np.dot(w, Zs[:, col])  # centre around weighted mean
        sc2 = np.dot(w, Zs[:, col] ** 2)
        scales[col] = np.sqrt(sc2)
        Zs[:, col] /= scales[col]

    # ── 2. Augmented-design lstsq ─────────────────────────────────────────────
    sqrt_w = np.sqrt(w)
    Zw = Zs * sqrt_w[:, None]
    yw = y * sqrt_w
    sqrt_P = np.sqrt(P)
    aug_Z = np.vstack([Zw, sqrt_P])
    aug_y = np.concatenate([yw, np.zeros(p)])
    theta_std, *_ = np.linalg.lstsq(aug_Z, aug_y, rcond=None)

    # Residuals in original y space, using standardised Z.
    # Using Zs @ theta_std is equivalent to the correct back-transformation
    # for all fit_intercept/penalize_controls combinations and avoids a
    # tricky sign error when fit_intercept=False (no intercept column to
    # absorb the centering shift of penalised columns).
    residuals = y - Zs @ theta_std

    # ── 3. Bread ─────────────────────────────────────────────────────────────
    G = Zw.T @ Zw  # weighted Gram in standardised coords
    A = G + P
    A_inv = np.linalg.inv(A)

    # ── 4. Meat: explicit Bartlett loop ───────────────────────────────────────
    scores = w[:, None] * Zs * residuals[:, None]  # (n, p) — in standardised coords
    S = scores.T @ scores
    for lag in range(1, n_lags + 1):
        bl = 1.0 - lag / (n_lags + 1)
        gamma = scores[lag:].T @ scores[:-lag]
        S += bl * (gamma + gamma.T)

    # ── 5. Sandwich in standardised coords ────────────────────────────────────
    n_eff = 1.0 / np.dot(w, w)
    correction = n_eff / (n_eff - p)
    V_std = correction * A_inv @ S @ A_inv

    # ── 6. Back to original units via scales ──────────────────────────────────
    # var(beta_j) = V_std[j,j] / scales[j]^2
    var_last = V_std[-1, -1] / scales[-1] ** 2
    return float(np.sqrt(var_last))


# ──────────────────────────────────────────────────────────────────────────────
# 1. Augmented-design sandwich reference
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n_controls", [0, 3])
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("penalize_controls", [True, False])
@pytest.mark.parametrize("n_lags", [0, 1, 5])
@pytest.mark.parametrize("ewma_halflife", [None, 10])
def test_augmented_design_reference(
    n_controls: int,
    fit_intercept: bool,
    penalize_controls: bool,
    n_lags: int,
    ewma_halflife: int | None,
) -> None:
    """get_se() agrees with an independently coded augmented-design sandwich."""
    if not penalize_controls and n_controls == 0:
        pytest.skip("penalize_controls=False with no controls is vacuous")

    T, W = 50, 50
    lambda_ = 0.4
    targets, factors, controls, _ = _panel(T, n_targets=2, n_controls=n_controls, seed=1)

    model = RollingOLS(
        window=W,
        min_periods=W,
        lambda_=lambda_,
        penalize_controls=penalize_controls,
        fit_intercept=fit_intercept,
        ewma_halflife=ewma_halflife,
        hac_lags=n_lags,
        dtype="float64",
    )
    result = model.fit_transform(factors, targets, controls=controls)

    # Reconstruct the window's weight vector
    if ewma_halflife is not None:
        lam = 0.5 ** (1.0 / ewma_halflife)
        raw = lam ** np.arange(W - 1, -1, -1, dtype=np.float64)
        w = raw / raw.sum()
    else:
        w = np.full(W, 1.0 / W)

    # Build the original-coordinate design matrix
    cols: list[pd.DataFrame] = []
    if controls is not None:
        cols.append(controls)
    cols.append(factors)
    X_df = pd.concat(cols, axis=1)

    if fit_intercept:
        Z = np.column_stack([np.ones(W), X_df.to_numpy()])
    else:
        Z = X_df.to_numpy()

    p = Z.shape[1]
    # Build penalty matching model._penalty_matrix
    P = np.zeros((p, p))
    slope_offset = int(fit_intercept)
    if penalize_controls and n_controls > 0:
        P[slope_offset : slope_offset + n_controls, slope_offset : slope_offset + n_controls] = (
            np.eye(n_controls) * lambda_
        )
    factor_pos = slope_offset + n_controls
    P[factor_pos, factor_pos] = lambda_

    # Compare for each target
    se_result = result.get_se("f")
    for col in targets.columns:
        y = targets[col].to_numpy()
        expected = _reference_ridge_hac_se(y, Z, w, P, n_lags, fit_intercept=fit_intercept)
        actual = float(se_result[col].dropna().iloc[-1])
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-11), (
            f"SE mismatch for target={col!r}, "
            f"n_controls={n_controls}, fit_intercept={fit_intercept}, "
            f"penalize_controls={penalize_controls}, n_lags={n_lags}, "
            f"ewma_halflife={ewma_halflife}: "
            f"got {actual}, expected {expected}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 2. Factor-rescaling equivariance
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("lambda_", [0.0, 0.5])
@pytest.mark.parametrize("scale", [2.0, 0.1, 100.0])
def test_rescaling_equivariance(lambda_: float, scale: float) -> None:
    """Scaling a factor by c scales beta by 1/c and SE by 1/c; t-stat is invariant."""
    T, W = 60, 60
    rng = np.random.default_rng(42)
    idx = pd.RangeIndex(T)
    f = rng.normal(size=T)
    y = 3.0 * f + 0.5 * rng.normal(size=T)
    factors = pd.DataFrame({"f": f}, index=idx)
    scaled_factors = pd.DataFrame({"f": f * scale}, index=idx)
    targets = pd.DataFrame({"a": y}, index=idx)

    kwargs = dict(window=W, min_periods=W, lambda_=lambda_, hac_lags=3, dtype="float64")
    r_orig = RollingOLS(**kwargs).fit_transform(factors, targets)
    r_scaled = RollingOLS(**kwargs).fit_transform(scaled_factors, targets)

    beta_orig = r_orig.get_beta("f")["a"].dropna()
    beta_scaled = r_scaled.get_beta("f")["a"].dropna()
    se_orig = r_orig.get_se("f")["a"].dropna()
    se_scaled = r_scaled.get_se("f")["a"].dropna()
    t_orig = r_orig.get_tstat("f")["a"].dropna()
    t_scaled = r_scaled.get_tstat("f")["a"].dropna()

    common = beta_orig.index.intersection(beta_scaled.index)

    # beta scales by 1/c: beta_scaled = beta_orig / c
    np.testing.assert_allclose(
        beta_orig.loc[common].values,
        beta_scaled.loc[common].values * scale,
        rtol=1e-10,
        atol=1e-12,
    )
    # SE scales by 1/c: se_scaled = se_orig / c
    np.testing.assert_allclose(
        se_orig.loc[common].values,
        se_scaled.loc[common].values * scale,
        rtol=1e-10,
        atol=1e-12,
    )
    # t-statistic is invariant
    np.testing.assert_allclose(
        t_orig.loc[common].values,
        t_scaled.loc[common].values,
        rtol=1e-10,
        atol=1e-12,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3. Direct augmented-design Ridge coefficient comparison
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n_controls", [0, 2])
@pytest.mark.parametrize("penalize_controls", [True, False])
def test_ridge_coefficients_via_lstsq(n_controls: int, penalize_controls: bool) -> None:
    """Ridge coefficients match an augmented-design lstsq solve to 1e-10."""
    if not penalize_controls and n_controls == 0:
        pytest.skip("penalize_controls=False with no controls is vacuous")

    T, W = 50, 50
    lambda_ = 0.3
    targets, factors, controls, _ = _panel(T, n_targets=2, n_controls=n_controls, seed=3)

    model = RollingOLS(
        window=W,
        min_periods=W,
        lambda_=lambda_,
        penalize_controls=penalize_controls,
        fit_intercept=True,
        dtype="float64",
    )
    result = model.fit_transform(factors, targets, controls=controls)

    # Build original-coordinate design matrix
    cols = [] if controls is None else [controls]
    cols.append(factors)
    X = pd.concat(cols, axis=1).to_numpy()
    y_all = targets.to_numpy()
    Z = np.column_stack([np.ones(W), X])
    p = Z.shape[1]

    # Build penalty
    P = np.zeros((p, p))
    if penalize_controls and n_controls > 0:
        P[1 : 1 + n_controls, 1 : 1 + n_controls] = np.eye(n_controls) * lambda_
    P[1 + n_controls, 1 + n_controls] = lambda_

    # Standardize penalized columns and solve via lstsq on augmented design
    Zs = Z.copy()
    means = np.zeros(p)
    scales_arr = np.ones(p)
    w = np.full(W, 1.0 / W)
    penalized_cols = np.flatnonzero(np.diag(P) > 0)
    for col in penalized_cols:
        means[col] = np.dot(w, Zs[:, col])
        Zs[:, col] -= means[col]
        sc2 = np.dot(w, Zs[:, col] ** 2)
        scales_arr[col] = np.sqrt(sc2)
        Zs[:, col] /= scales_arr[col]

    sqrt_w = np.sqrt(w)
    Zw = Zs * sqrt_w[:, None]
    yw_all = y_all * sqrt_w[:, None]
    sqrt_P = np.sqrt(P)
    aug_Z = np.vstack([Zw, sqrt_P])
    aug_y_all = np.vstack([yw_all, np.zeros((p, y_all.shape[1]))])

    theta_std, *_ = np.linalg.lstsq(aug_Z, aug_y_all, rcond=None)
    theta = theta_std / scales_arr[:, None]
    theta[0] -= means[1:] @ theta[1:]

    # Compare betas (last slope) and intercepts
    expected_beta = theta[1 + n_controls, :]
    expected_intercept = theta[0, :]

    for i, col in enumerate(targets.columns):
        actual_beta = float(result.get_beta("f")[col].dropna().iloc[-1])
        actual_intercept = float(result.get_intercept("f")[col].dropna().iloc[-1])
        assert actual_beta == pytest.approx(expected_beta[i], rel=1e-10, abs=1e-12), (
            f"beta mismatch for target={col!r}: {actual_beta} vs {expected_beta[i]}"
        )
        assert actual_intercept == pytest.approx(expected_intercept[i], rel=1e-10, abs=1e-12), (
            f"intercept mismatch for target={col!r}: {actual_intercept} vs {expected_intercept[i]}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 4. statsmodels WLS comparison for the unpenalized path
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n_lags", [0, 3])
@pytest.mark.parametrize("n_controls", [0, 2])
def test_statsmodels_wls_hac(n_lags: int, n_controls: int) -> None:
    """Unpenalized EWMA path matches statsmodels.WLS with HAC covariance."""
    import statsmodels.api as sm

    T, W, halflife = 60, 60, 15
    targets, factors, controls, _ = _panel(T, n_targets=1, n_controls=n_controls, seed=5)

    model = RollingOLS(
        window=W,
        min_periods=W,
        lambda_=0.0,
        ewma_halflife=halflife,
        hac_lags=n_lags,
        dtype="float64",
    )
    result = model.fit_transform(factors, targets.iloc[:, :1], controls=controls)
    rols_se = float(result.get_se("f").iloc[-1, 0])

    # Build weight vector (match rOLS internal: decay from present to past, renorm)
    lam = 0.5 ** (1.0 / halflife)
    raw = lam ** np.arange(W - 1, -1, -1, dtype=np.float64)
    w = raw / raw.sum()
    n_eff = 1.0 / np.dot(w, w)

    # Build design matrix
    X_cols = [] if controls is None else [controls.to_numpy()]
    X_cols.append(factors.to_numpy())
    X = np.hstack(X_cols)
    endog = targets.iloc[:, 0].to_numpy()
    exog = sm.add_constant(X, has_constant="add")
    k = exog.shape[1]

    # statsmodels WLS with use_correction=False gives the *asymptotic* sandwich
    # V (no small-sample correction).  rOLS always applies n_eff/(n_eff-k);
    # for EWMA weights n_eff < W so the two corrections differ.  We validate
    # the raw sandwich components are the same by undoing the rOLS correction.
    fit_wls = sm.WLS(endog, exog, weights=W * w).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": n_lags, "use_correction": False},
    )
    sm_se_raw = float(fit_wls.bse[-1])  # asymptotic (uncorrected) SE

    # rOLS correction factor: n_eff/(n_eff-k); undo to recover raw SE
    rols_se_raw = rols_se * np.sqrt((n_eff - k) / n_eff)

    assert rols_se_raw == pytest.approx(sm_se_raw, rel=1e-8, abs=1e-11), (
        f"raw sandwich mismatch: rOLS (correction removed)={rols_se_raw:.6e} "
        f"vs statsmodels (uncorrected)={sm_se_raw:.6e} "
        f"(n_lags={n_lags}, n_controls={n_controls})"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 5. Independently coded Bartlett loop reference
# ──────────────────────────────────────────────────────────────────────────────


def _newey_west_reference(
    y: np.ndarray,
    Z: np.ndarray,
    w: np.ndarray,
    P: np.ndarray,
    n_lags: int,
) -> float:
    """Newey-West SE from Newey & West (1987), written from the definition.

    Deliberately slow and literal: no vectorization, no production code shared.
    Returns the SE for the last regressor (the factor of interest).
    """
    n, p = Z.shape
    n_eff = 1.0 / np.dot(w, w)

    # Standardize: centre only when fit_intercept=True (here always True —
    # Z always has an intercept column in this function's callers).
    # Production code only centres penalised columns when fit_intercept=True
    # (estimators.py:328-330).
    penalized = np.flatnonzero(np.diag(P) > 0)
    Zs = Z.copy()
    sc = np.ones(p)
    for col in penalized:
        Zs[:, col] -= np.dot(w, Zs[:, col])  # fit_intercept=True → centre
        sc[col] = np.sqrt(np.dot(w, Zs[:, col] ** 2))
        Zs[:, col] /= sc[col]

    sqrt_w = np.sqrt(w)
    Zw = Zs * sqrt_w[:, None]
    G = Zw.T @ Zw
    A = G + P
    A_inv = np.linalg.inv(A)

    # Ridge coefficient in standardised coordinates via augmented-design lstsq.
    aug_Z = np.vstack([Zw, np.sqrt(P)])
    aug_y = np.concatenate([y * sqrt_w, np.zeros(p)])
    theta_std, *_ = np.linalg.lstsq(aug_Z, aug_y, rcond=None)
    # Residuals in original y space using standardised Z; avoids tricky
    # back-transformation when fit_intercept=False (no intercept to absorb
    # the centering shift of penalised columns).
    resid = y - Zs @ theta_std

    # Scores in standardised coordinates: g_s = w_s * z_s_std * u_s
    G_mat = np.zeros((p, p))
    for s in range(n):
        g_s = w[s] * Zs[s] * resid[s]
        G_mat += np.outer(g_s, g_s)
    for lag in range(1, n_lags + 1):
        bl = 1.0 - lag / (n_lags + 1)
        for s in range(lag, n):
            g_s = w[s] * Zs[s] * resid[s]
            g_sl = w[s - lag] * Zs[s - lag] * resid[s - lag]
            G_mat += bl * (np.outer(g_s, g_sl) + np.outer(g_sl, g_s))

    V_std = (n_eff / (n_eff - p)) * A_inv @ G_mat @ A_inv
    var_last = V_std[-1, -1] / sc[-1] ** 2
    return float(np.sqrt(var_last))


@pytest.mark.parametrize("lambda_", [0.0, 0.3])
@pytest.mark.parametrize("n_lags", [0, 2])
def test_newey_west_reference_loop(lambda_: float, n_lags: int) -> None:
    """rolling_hac_se matches a hand-coded Newey-West loop on a single window."""
    T, W = 30, 30
    rng = np.random.default_rng(99)
    idx = pd.RangeIndex(T)
    f = rng.normal(size=T)
    c = rng.normal(size=T)
    y = 2.0 * f - c + 0.5 * rng.normal(size=T)
    factors = pd.DataFrame({"f": f}, index=idx)
    controls = pd.DataFrame({"c": c}, index=idx)
    targets = pd.DataFrame({"a": y}, index=idx)

    design = pd.concat([controls, factors], axis=1)
    p_full = 1 + design.shape[1]  # intercept + control + factor
    P = np.zeros((p_full, p_full))
    P[1, 1] = lambda_  # control
    P[2, 2] = lambda_  # factor

    se_prod = rolling_hac_se(
        targets,
        design,
        window=W,
        min_periods=W,
        expanding=False,
        n_lags=n_lags,
        penalty=P,
    )
    actual = float(se_prod.iloc[-1, 0])

    # Reference
    Z = np.column_stack([np.ones(W), c, f])
    w = np.full(W, 1.0 / W)
    expected = _newey_west_reference(y, Z, w, P, n_lags)

    assert actual == pytest.approx(expected, rel=1e-7, abs=1e-10), (
        f"Newey-West reference mismatch at lambda_={lambda_}, n_lags={n_lags}: "
        f"production={actual:.8e}, reference={expected:.8e}"
    )
