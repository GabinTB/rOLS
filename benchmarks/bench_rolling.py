"""Reproducible wall-time and memory benchmarks for rolling panel estimation."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import inspect
import json
import platform
import resource
import sys
import time
import tracemalloc
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from rols import RollingOLS

SCHEMA_VERSION = 1
WINDOW = 252
NAN_PATTERNS = ("clean", "structural", "scattered")


@dataclass(frozen=True)
class PanelSize:
    """Dimensions for one named benchmark panel."""

    T: int
    N: int
    K: int
    q: int
    slow: bool = False


@dataclass(frozen=True)
class BenchmarkConfig:
    """Estimator settings varied by the benchmark grid."""

    lambda_: float
    ewma_halflife: int | None
    mode: str = "batched"


SIZE_GRID = {
    "tiny": PanelSize(T=500, N=20, K=3, q=2),
    "small": PanelSize(T=1260, N=100, K=5, q=3),
    "medium": PanelSize(T=2520, N=500, K=20, q=3, slow=True),
    "large": PanelSize(T=5040, N=2300, K=50, q=3, slow=True),
}

CONFIGS = tuple(
    BenchmarkConfig(lambda_=lambda_, ewma_halflife=halflife)
    for lambda_ in (0.0, 1e-3)
    for halflife in (None, 63)
)


def make_panel(
    T: int,
    N: int,
    K: int,
    q: int,
    nan_pattern: str,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a deterministic benchmark panel with a shared date index.

    ``nan_pattern`` is one of ``clean``, ``structural``, or ``scattered``.
    Structural missingness gives approximately 15% of assets one contiguous
    lifespan that starts late or ends early. Scattered missingness removes 1%
    of target entries independently.
    """
    if T <= 0 or N <= 0 or K <= 0 or q < 0:
        raise ValueError("T, N, and K must be positive and q must be non-negative")
    if nan_pattern not in NAN_PATTERNS:
        raise ValueError(f"nan_pattern must be one of {NAN_PATTERNS}")

    rng = np.random.default_rng(seed)
    index = pd.date_range("2000-01-03", periods=T, freq="B")
    factors_array = rng.normal(size=(T, K))
    controls_array = rng.normal(size=(T, q))
    factor_loadings = rng.normal(scale=0.4, size=(K, N))
    control_loadings = rng.normal(scale=0.25, size=(q, N))
    # NumPy/Accelerate can emit spurious floating-point matmul warnings on
    # finite inputs. Validate the output explicitly instead of leaking them.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        targets_array = factors_array @ factor_loadings
        if q:
            targets_array += controls_array @ control_loadings
    if not np.isfinite(targets_array).all():
        raise FloatingPointError("panel generation produced non-finite values")
    targets_array += rng.normal(size=(T, N))

    if nan_pattern == "structural":
        n_structural = max(1, round(0.15 * N))
        affected = rng.choice(N, size=n_structural, replace=False)
        for position, asset in enumerate(affected):
            if position % 2 == 0:
                start = int(rng.integers(max(1, T // 20), max(2, T // 3)))
                targets_array[:start, asset] = np.nan
            else:
                stop = int(rng.integers(max(1, 2 * T // 3), T))
                targets_array[stop:, asset] = np.nan
    elif nan_pattern == "scattered":
        n_missing = max(1, round(0.01 * T * N))
        missing = rng.choice(T * N, size=n_missing, replace=False)
        targets_array.ravel()[missing] = np.nan

    targets = pd.DataFrame(targets_array, index=index, columns=[f"asset_{i}" for i in range(N)])
    factors = pd.DataFrame(factors_array, index=index, columns=[f"factor_{i}" for i in range(K)])
    controls = pd.DataFrame(
        controls_array,
        index=index,
        columns=[f"control_{i}" for i in range(q)],
    )
    return targets, factors, controls


def _rss_bytes() -> int:
    """Return the process high-water RSS in bytes on macOS and Linux."""
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss if sys.platform == "darwin" else rss * 1024


def _measure(call: Callable[[], Any]) -> tuple[Any, dict[str, float | int]]:
    """Measure one call's wall time, Python allocation peak, and process RSS peak."""
    gc.collect()
    rss_before = _rss_bytes()
    tracemalloc.start()
    started = time.perf_counter()
    try:
        value = call()
        wall_seconds = time.perf_counter() - started
        _, python_peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    peak_rss_bytes = _rss_bytes()
    measurement: dict[str, float | int] = {
        "wall_seconds": wall_seconds,
        "python_peak_bytes": int(python_peak_bytes),
        "peak_rss_bytes": peak_rss_bytes,
        "rss_increase_bytes": max(0, peak_rss_bytes - rss_before),
    }
    return value, measurement


def _materialized_size_bytes(value: Any, seen: set[int] | None = None) -> int:
    """Count pandas and NumPy buffers reachable from a materialized result."""
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return 0
    seen.add(value_id)

    if isinstance(value, pd.DataFrame):
        return int(value.memory_usage(index=True, deep=True).sum())
    if isinstance(value, pd.Series):
        return int(value.memory_usage(index=True, deep=True))
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(_materialized_size_bytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return sum(_materialized_size_bytes(item, seen) for item in value)
    if hasattr(value, "__dict__"):
        return _materialized_size_bytes(vars(value), seen)
    return 0


def _model_kwargs(config: BenchmarkConfig) -> dict[str, Any]:
    """Build kwargs supported by both the current estimator and v0.2.1."""
    kwargs: dict[str, Any] = {
        "window": WINDOW,
        "min_periods": WINDOW,
        "lambda_": config.lambda_,
        "ewma_halflife": config.ewma_halflife,
        "hac_lags": 5,
        "warn_singular": False,
    }
    if "mode" in inspect.signature(RollingOLS).parameters:
        kwargs["mode"] = config.mode
    return kwargs


def run_case(
    size_name: str,
    nan_pattern: str,
    config: BenchmarkConfig,
    seed: int = 0,
) -> dict[str, Any]:
    """Run one grid cell and return a JSON-serializable benchmark record."""
    if size_name not in SIZE_GRID:
        raise ValueError(f"unknown size {size_name!r}")
    size = SIZE_GRID[size_name]
    targets, factors, controls = make_panel(
        size.T,
        size.N,
        size.K,
        size.q,
        nan_pattern,
        seed=seed,
    )
    model = RollingOLS(**_model_kwargs(config))
    controls_arg = controls if size.q else None

    _, fit_measurement = _measure(lambda: model.fit(factors, controls=controls_arg))
    result, transform_measurement = _measure(lambda: model.transform(targets))
    factor = factors.columns[0]
    accessors: dict[str, dict[str, float | int] | None] = {}
    for name in ("get_beta", "get_r2", "get_residuals", "get_se"):
        accessor = getattr(result, name, None)
        if accessor is None:
            accessors[name] = None
            continue
        _, accessors[name] = _measure(lambda accessor=accessor: accessor(factor))

    total_wall_seconds = fit_measurement["wall_seconds"] + transform_measurement["wall_seconds"]
    total_wall_seconds += sum(
        measurement["wall_seconds"] for measurement in accessors.values() if measurement is not None
    )
    return {
        "size": size_name,
        "dimensions": asdict(size),
        "nan_pattern": nan_pattern,
        "config": asdict(config),
        "measurements": {
            "fit": fit_measurement,
            "transform": transform_measurement,
            "accessors": accessors,
            "total_wall_seconds": total_wall_seconds,
        },
        "output_size_bytes": _materialized_size_bytes(result),
    }


def run_grid(
    sizes: Iterable[str],
    nan_patterns: Iterable[str] = NAN_PATTERNS,
    configs: Iterable[BenchmarkConfig] = CONFIGS,
    seed: int = 0,
    output: Path | None = None,
    source_label: str = "current",
) -> dict[str, Any]:
    """Run selected grid cells, checkpointing and resuming when output is set."""
    if output is not None and output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"cannot resume schema version in {output}")
    else:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "metadata": {
                "source": source_label,
                "rols_version": importlib.metadata.version("rols"),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "window": WINDOW,
                "seed": seed,
            },
            "records": [],
        }
    completed = {_record_key(record) for record in payload["records"]}
    for size_name in sizes:
        for nan_pattern in nan_patterns:
            for config in configs:
                key = (
                    size_name,
                    nan_pattern,
                    config.lambda_,
                    config.ewma_halflife,
                    config.mode,
                )
                if key in completed:
                    continue
                record = run_case(size_name, nan_pattern, config, seed=seed)
                payload["records"].append(record)
                completed.add(key)
                if output is not None:
                    _write_json(output, payload)
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    config = record["config"]
    return (
        record["size"],
        record["nan_pattern"],
        config["lambda_"],
        config["ewma_halflife"],
        config.get("mode", "batched"),
    )


def _wall_metrics(record: dict[str, Any]) -> dict[str, float | None]:
    measurements = record["measurements"]
    metrics: dict[str, float | None] = {
        "fit": measurements["fit"]["wall_seconds"],
        "transform": measurements["transform"]["wall_seconds"],
        "total": measurements["total_wall_seconds"],
    }
    metrics.update(
        {
            name: None if value is None else value["wall_seconds"]
            for name, value in measurements["accessors"].items()
        }
    )
    return metrics


def compare_payloads(current: dict[str, Any], baseline: dict[str, Any]) -> bool:
    """Print current/baseline ratios and return whether the 3x tripwire passed."""
    baseline_by_key = {_record_key(record): record for record in baseline["records"]}
    header = f"{'case':<58} {'metric':<14} {'current':>12} {'baseline':>12} {'ratio':>9}"
    print(header)
    print("-" * len(header))
    passed = True
    for record in current["records"]:
        key = _record_key(record)
        baseline_record = baseline_by_key.get(key)
        if baseline_record is None:
            continue
        case = "/".join(str(part) for part in key)
        current_metrics = _wall_metrics(record)
        baseline_metrics = _wall_metrics(baseline_record)
        for metric, current_value in current_metrics.items():
            baseline_value = baseline_metrics.get(metric)
            if current_value is None or baseline_value is None or baseline_value <= 0:
                ratio_text = "null"
            else:
                ratio = current_value / baseline_value
                ratio_text = f"{ratio:.2f}x"
                if record["size"] == "medium" and metric == "total" and ratio > 3.0:
                    passed = False
            print(
                f"{case:<58} {metric:<14} "
                f"{_format_seconds(current_value):>12} "
                f"{_format_seconds(baseline_value):>12} {ratio_text:>9}"
            )
    return passed


def _format_seconds(value: float | None) -> str:
    return "null" if value is None else f"{value:.6f}"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", action="append", choices=SIZE_GRID)
    parser.add_argument("--nan-pattern", action="append", choices=NAN_PATTERNS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source-label", default="current")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.size:
        sizes = args.size
    elif args.compare:
        sizes = ["tiny", "small", "medium"]
    else:
        sizes = ["tiny", "small"]
    nan_patterns = args.nan_pattern or list(NAN_PATTERNS)
    payload = run_grid(
        sizes=sizes,
        nan_patterns=nan_patterns,
        seed=args.seed,
        output=args.output,
        source_label=args.source_label,
    )
    if args.output is not None:
        _write_json(args.output, payload)
    if args.compare is None:
        if args.output is None:
            print(json.dumps(payload, indent=2))
        else:
            print(f"Wrote {len(payload['records'])} records to {args.output}")
        return 0

    baseline = json.loads(args.compare.read_text(encoding="utf-8"))
    return 0 if compare_payloads(payload, baseline) else 1


@pytest.mark.parametrize("size_name", ["tiny", "small"])
@pytest.mark.parametrize("nan_pattern", NAN_PATTERNS)
@pytest.mark.parametrize("config", CONFIGS)
def test_default_benchmark_grid(benchmark, size_name, nan_pattern, config):
    """pytest-benchmark entry point for the default tiny and small grid."""
    benchmark.pedantic(
        run_case,
        args=(size_name, nan_pattern, config),
        rounds=1,
        iterations=1,
    )


@pytest.mark.slow
@pytest.mark.parametrize("size_name", ["medium", "large"])
@pytest.mark.parametrize("nan_pattern", NAN_PATTERNS)
@pytest.mark.parametrize("config", CONFIGS)
def test_slow_benchmark_grid(benchmark, size_name, nan_pattern, config):
    """Explicit slow benchmark entry point for the medium and large grid."""
    benchmark.pedantic(
        run_case,
        args=(size_name, nan_pattern, config),
        rounds=1,
        iterations=1,
    )


if __name__ == "__main__":
    raise SystemExit(main())
