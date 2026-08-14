# rOLS benchmark harness

This harness measures rolling-panel wall time, Python allocation peaks, process
peak RSS, and materialized result size. It covers clean data, realistic
structural asset histories, and scattered missingness across four panel sizes.

The checked-in v0.2.1 baseline is a **speed reference only**. v0.2.1's estimates
are known to be statistically incorrect. Never use its outputs as a correctness
reference or alter corrected estimator behaviour to reproduce them.

The checked-in capture contains all tiny, small, and medium cases and one of 12
large cases. That large case took 637.5 seconds, reached 5.43 GB peak RSS, and
materialized an 18.74 GB result. The next case was interrupted during v0.2.1
`transform()`. The JSON metadata records this limit explicitly. Rerun the
command below to resume the remaining large cases on a machine suitable for an
unbounded target-scale capture.

## Running the grid

Tiny and small run by default:

```bash
uv run python -m benchmarks.bench_rolling
uv run python -m benchmarks.bench_rolling --size small
```

The equivalent pytest-benchmark grid is:

```bash
uv run pytest benchmarks/bench_rolling.py -m "not slow"
```

Medium and large are slow and must be selected explicitly:

```bash
uv run python -m benchmarks.bench_rolling --size medium --output /tmp/medium.json
uv run python -m benchmarks.bench_rolling --size large --output /tmp/large.json
uv run pytest benchmarks/bench_rolling.py -m slow
```

Each record is checkpointed when `--output` is supplied. Rerunning the same
command resumes the file and skips completed grid cells. This matters for the
large grid, where a later failure should not discard earlier measurements.

## Capturing v0.2.1

Use the exact v0.2.1 commit in an isolated worktree, copy this benchmark package
into that scratch checkout, and run all four sizes there. Existing records are
retained, so the same command resumes an interrupted capture:

```bash
git worktree add /tmp/rols-v021 6c3b836
cp -R benchmarks /tmp/rols-v021/
cd /tmp/rols-v021
uv run python -m benchmarks.bench_rolling \
  --size tiny --size small --size medium --size large \
  --source-label v0.2.1@6c3b836 \
  --output benchmarks/baseline_v0.2.1.json
```

Copy the resulting JSON back to the corrected checkout. Accessors absent from
v0.2.1 are represented as `null` while retaining the same record schema.

## Comparing

```bash
uv run python -m benchmarks.bench_rolling \
  --compare benchmarks/baseline_v0.2.1.json
```

Comparison mode runs tiny, small, and medium by default, prints wall-time ratios
for every common measurement, and exits non-zero when a medium case's total
measured time exceeds 3x baseline. This is a regression tripwire, not the final
performance target.
