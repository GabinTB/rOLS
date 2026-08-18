# rOLS performance guide

This document explains where time and memory actually go, when the fast paths
apply, and reports measured numbers from the `benchmarks/` harness (task 10).
It is a companion to [`docs/SPECIFICATION.md`](SPECIFICATION.md), which
defines *what* is computed; this document is about the *cost* of computing it.

All numbers below were captured on the machine and Python build recorded in
`benchmarks/baseline_v0.2.1.json`'s metadata, using
`uv run python -m benchmarks.bench_rolling`. They are wall-clock, single-run
measurements on a shared development machine, not a controlled benchmark
environment — treat exact figures as indicative of magnitude and direction,
not as a guaranteed SLA. Re-run the harness on your own hardware for numbers
you plan to rely on.

---

## Cost model

For one endpoint `t` and one selected model (one factor in batched mode, all
`K` factors in joint mode), the dominant costs are:

1. **Factorizing the design** `[1, controls, ...]` on the window's
   complete-case rows — `O(W · p²)` for `p` design columns, `W` the window
   length (task 07's QR/augmented-QR factorization, chosen over the normal
   equations specifically so this step does not square the condition number).
2. **Solving for every target sharing that design** as a block of right-hand
   sides — one BLAS-3 GEMM/triangular-solve per (endpoint, pattern) instead of
   one per (endpoint, target). This is where pattern grouping and the FWL fast
   path both pay off, and where nearly all the wall-clock time in a large
   panel actually lives.
3. **HAC inference**, when requested, per (endpoint, target): a Bartlett-
   weighted sandwich over the window's complete-case scores. Streamed one
   endpoint at a time (`rols/estimators.py::rolling_hac_se`) rather than
   materializing scores for the whole panel.

The rolling window views themselves (`_make_windows`) are zero-copy —
`numpy.lib.stride_tricks.as_strided` — so windowing is not a cost center; the
GEMMs and factorizations are.

---

## Pattern grouping

Per-target complete-case masking means two targets can, in principle, need two
different designs even within the same window. Grouping targets by their
*exact* complete-case mask and factorizing once per distinct mask — instead of
once per target — is what makes panel-scale problems tractable:

- **`clean`** (no missingness): one pattern, the trivial case. Grouping adds a
  small fixed hashing overhead for no benefit, since there was only ever one
  group.
- **`structural`** (assets entering/leaving the index, ~15% of assets do so
  mid-sample): a handful of distinct patterns dominated by one "all present"
  group. This is the realistic case an index panel actually produces, and the
  one grouping was built for.
- **`scattered`** (missingness spread uniformly at random): every target ends
  up with a near-unique pattern, so grouping degenerates to the per-target
  path. Correct, but the grouping machinery buys nothing here — measure your
  own panel's missingness shape before assuming grouping helps it.

Measured small-grid transform time by pattern (`T=1260, N=100, K=5, q=3`,
`window=252`, OLS, no EWMA):

| Pattern | transform() | vs `clean` |
|---|---|---|
| `clean` | 628.7 ms | 1.0x |
| `structural` | 695.8 ms | 1.1x |
| `scattered` | 1517.5 ms | 2.4x |

`scattered` costing ~2.4x `clean` — rather than `N`x — is the grouping win:
without it, `scattered`'s cost would scale with the number of *targets*, not
the number of *distinct patterns*.

---

## When the FWL fast path applies

`result._path` (private, for tests) is one of `"fwl"`, `"joint"`, or
`"per-target"`. The FWL fast path — sharing one `[1, controls]` projection and
one GEMM across every factor and target — applies only when:

- `lambda_ == 0` (OLS). Ridge does not commute with FWL residualization (three
  independent audit reviewers confirmed this independently), so any
  `lambda_ > 0` **always** routes to the direct joint solve, never FWL.
- Controls are present. With no controls, the nuisance projection is trivial
  and the routing collapses accordingly.

This is why `lambda_ > 0` is measurably more expensive than `lambda_ == 0`, and
why the gap is *not* a fixed constant — it is the FWL fast path being
unavailable, not merely "Ridge does a bit more arithmetic":

| Pattern | `lambda_=0` | `lambda_=1e-3` | ratio |
|---|---|---|---|
| `clean` | 628.7 ms | 2001.0 ms | 3.2x |
| `structural` | 695.8 ms | 3629.8 ms | 5.2x |
| `scattered` | 1517.5 ms | 5694.9 ms | 3.8x |

Note that v0.2.1 shows no comparable slowdown under `lambda_ > 0` in its own
baseline capture. That is because v0.2.1's Ridge was not a real joint solve at
all — findings F4 and F5 established that `lambda_` had no effect without
controls and penalized only a residualization step with controls, never the
reported factor coefficient. v0.2.1's Ridge numbers are therefore not a valid
speed comparison for this cost: the current implementation is doing
substantially more real work because it is now doing the *correct* work. See
`benchmarks/README.md` for the general caveat that v0.2.1 is a speed
reference, never a correctness one.

---

## Batched vs joint

`mode="joint"`'s design has `K + len(controls) + 1` columns and is solved once
per window; `mode="batched"` solves each factor separately but, under OLS,
shares the controls-only projection and does one GEMM across all `K` factors
via the FWL fast path (see above). Which is cheaper depends entirely on
`lambda_`:

**Under OLS (`lambda_ == 0`)**, batched's shared GEMM wins, increasingly so as
`K` or panel size grows:

| `T` | `N` | `K` | `window` | batched | joint | joint / batched |
|---|---|---|---|---|---|---|
| 1260 | 100 | 5 | 60 | 0.258 s | 0.173 s | 0.67x |
| 1260 | 100 | 20 | 60 | 0.342 s | 0.407 s | 1.19x |
| 1260 | 100 | 50 | 60 | 0.465 s | 1.057 s | 2.27x |
| 2520 | 500 | 3 | 252 | 3.405 s | 5.766 s | 1.69x |
| 2520 | 500 | 20 | 252 | 3.963 s | 15.848 s | 4.00x |
| 2520 | 500 | 50 | 252 | 4.664 s | 37.996 s | 8.15x |

At small `K` and short windows, joint can be marginally cheaper (one small
solve beats K small solves plus GEMM setup); at panel scale — larger windows,
more targets, more factors — joint's per-window solve cost grows faster than
batched's shared-GEMM cost, and batched pulls decisively ahead. There is no
robust `K ≥ 5` crossover independent of scale; the crossover point itself
depends on window length and target count. Benchmark your own configuration
before assuming either direction.

**Under Ridge (`lambda_ > 0`)**, the FWL fast path is unavailable (see above),
so batched degrades to `K` separate joint-equivalent solves while joint mode
is exactly *one* such solve — joint is then substantially cheaper:

| `T` | `N` | `K` | `window` | `lambda_` | batched | joint | joint / batched |
|---|---|---|---|---|---|---|---|
| 1260 | 100 | 20 | 60 | 1e-3 | 2.412 s | 0.504 s | 0.21x |

So: default to batched for OLS screening at panel scale; prefer joint whenever
factors are correlated (statistical correctness) or `lambda_ > 0` (also
faster there). Neither mode is a strictly dominant default — see
[the README](../README.md#batched-vs-joint-mode) for the decision in plain
terms.

---

## Memory

`estimate_memory()` computes the answer for your actual shapes before you fit
anything; the table below shows it across the benchmark harness's standard
grid (`window=252`, `cache_size=1`):

| Grid | `T` | targets | factors | controls | `estimate_memory()["total"]` |
|---|---|---|---|---|---|
| `tiny` | 500 | 20 | 3 | 2 | 0.9 MB |
| `small` | 1260 | 100 | 5 | 3 | 15.3 MB |
| `medium` | 2520 | 500 | 20 | 3 | 454.5 MB |
| `large` | 5040 | 2300 | 50 | 3 | 9741.4 MB (9.74 GB) |

Breaking down the `large` grid's total by component (`cache_size=1`):

| Component | Bytes | What it is |
|---|---|---|
| `betas` | 4636.8 MB | Persistent, all 50 factors, materialized at `transform()` |
| `intercepts` | 4636.8 MB | Same shape as `betas`, same reason |
| `n_used` | 92.7 MB | Per-pattern complete-case counts |
| `pattern_statistics` | 280.2 MB | Sufficient statistics for lazy R²/residual derivation |
| `retained_inputs` | 94.9 MB | The input panel itself, retained for lazy recomputation |
| `on_demand_per_frame` | 92.7 MB | One accessor call's full-index output (e.g. one `get_beta(f)`) |
| `on_demand_cache_bytes` | 92.7 MB × `cache_size` | What the bounded lazy cache retains at once |

`betas`/`intercepts` dominate because they are materialized eagerly for every
factor at `transform()` — that cost is unavoidable and roughly `2 × K ×
on_demand_per_frame`. Everything *lazy* (`get_r2`, `get_residuals`, `get_se`,
`get_partial_r2`) costs `on_demand_per_frame` **per factor retained**, which is
where uncontrolled accumulation gets expensive: computing and keeping 4
different lazy quantities for all 50 factors without releasing any of them
costs roughly `4 × 50 × 92.7 MB ≈ 18 GB` on this grid, on top of the 9.74 GB
above.

This is exactly what `cache_size` and the `iter_*` accessors exist to bound:

```python
for factor, se in result.iter_se():
    process(factor, se)   # se is released once `process` returns and the
                           # loop advances — no O(K) accumulation
```

`estimate_every` reduces every per-frame and pattern-statistics cost roughly
proportionally to how many endpoints it skips — cutting cadence by 5x cuts
these costs by roughly 5x too, independent of the memory savings on `betas`
and `intercepts`.

Missing values in a factor split its sufficient statistics into
factor-specific complete-case patterns (see Pattern grouping, above) and can
increase `pattern_statistics` beyond the clean-data figure — more distinct
patterns means more retained per-pattern state.

---

## Rank-1 updating: considered and rejected

A natural-looking optimization for a rolling window is a rank-1 Woodbury /
Sherman-Morrison update — add the incoming row, downdate the outgoing row,
`O(p²)` per step instead of refactorizing the whole window at `O(W·p²)`. This
was proposed during the audit (finding F21) and deliberately **not**
implemented.

The reason is numerical, not a missed opportunity: Björck, Park & Elden (SIAM
J. Matrix Anal. Appl. 15, 1994) show that R-only downdating "may not recover
accuracy after an ill-conditioned problem has occurred" — one poorly
conditioned window along the way poisons every subsequent update, and nothing
short of periodic refactorization recovers it. `statsmodels`' own rolling
implementation carries a `reset` parameter for exactly this failure mode. Task
07 already replaced the normal-equations solve specifically because it was
*not* detecting ill-conditioning reliably; adding an update scheme whose
accuracy depends on never hitting an ill-conditioned window along a
potentially thousand-step rolling path would reopen that exact problem.

More decisively: per the cost model above, the factorization step is not this
library's bottleneck at panel scale — the shared GEMM across targets is. A
rank-1 update would only reduce a cost that is already a minority of the
total, in exchange for a real correctness risk over long rolling paths. This
is recorded here so the question is not silently reopened without new
information (e.g. a periodic-refactorization scheme with a proven accuracy
bound, benchmarked against the current factorization cost specifically, not
assumed to help).

---

## Reproducing these numbers

```bash
# Fast grids — CI-safe
uv run python -m benchmarks.bench_rolling --size tiny
uv run python -m benchmarks.bench_rolling --size small --with-se

# Regression gate and target scale — slow, explicit opt-in
uv run python -m benchmarks.bench_rolling --size medium --output /tmp/medium.json
uv run python -m benchmarks.bench_rolling --size large --output /tmp/large.json

# Against the v0.2.1 speed baseline (tiny/small/medium by default)
uv run python -m benchmarks.bench_rolling --compare benchmarks/baseline_v0.2.1.json
```

`--compare` exits non-zero if any `medium` case exceeds 3x the v0.2.1
baseline's wall time — a regression tripwire, not a final performance target.
Ridge (`lambda_ > 0`) cases are expected to run slower than v0.2.1, for the
reason explained above; the tripwire and any manual comparison should be read
with that in mind rather than treated as a flat pass/fail. `benchmarks/README.md`
has the full harness documentation, including how the `large`-grid baseline
was captured and its known-incomplete coverage: the checked-in capture has
all `tiny`/`small`/`medium` cases and one of twelve `large` cases (637.5 s,
5.43 GB peak RSS, an 18.74 GB materialized result), and the next `large` case
was interrupted during v0.2.1's own `transform()` — a data point for why
memory, not just time, is the real constraint at target scale.
