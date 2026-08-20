# rOLS Test Suite

The rOLS test suite validates the vectorized implementations against a slow, scalar reference implementation (the oracle) and verifies statistical and numerical invariants.

## Suite Overview

- `oracle.py` — scalar window-by-window reference for differential testing
- `test_differential.py` — parametrised sweep over controls, intercept, mode, lambda_, EWMA, window type, cadence, NaN pattern; every quantity checked against the oracle
- `test_hac_reference.py` — augmented-design Ridge sandwich, factor rescaling equivariance, lstsq Ridge coefficient comparison, statsmodels WLS, independently coded Bartlett loop
- `test_fwl_conditioning.py` — rank deficiency, near-duplicate columns, near-collinear factor/control, `cond_warn_threshold` boundary
- `test_invariants.py` — coherence identity, R² reproducibility, scale equivariance, location invariance, permutation invariance, cross-asset isolation, chunk/cadence/lazy invariance, FWL-vs-joint path agreement
- `test_edge_cases.py` — degenerate inputs, boundary conditions
- `test_lazy_memory.py` — memory-budget assertions

## Execution

Most tests are fast and run by default:

```bash
uv run pytest
```

To run the complete suite, including computationally expensive parametric sweeps and large-grid assertions, enable the slow tests:

```bash
uv run pytest -m slow
```

To run the full differential sweep in isolation:

```bash
uv run pytest tests/test_differential.py -m slow
```
