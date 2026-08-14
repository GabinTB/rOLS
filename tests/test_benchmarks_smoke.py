"""Smoke tests for the benchmark harness and record schema."""

import numpy as np
import pandas as pd
import pytest

from benchmarks.bench_rolling import (
    SIZE_GRID,
    BenchmarkConfig,
    compare_payloads,
    make_panel,
    run_case,
)


@pytest.mark.parametrize("nan_pattern", ["clean", "structural", "scattered"])
def test_panel_generator_is_deterministic_and_aligned(nan_pattern):
    first = make_panel(80, 20, 3, 2, nan_pattern, seed=10)
    second = make_panel(80, 20, 3, 2, nan_pattern, seed=10)

    for first_frame, second_frame in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(first_frame, second_frame)
        pd.testing.assert_index_equal(first[0].index, first_frame.index)


def test_structural_assets_have_contiguous_lifespans():
    targets, _, _ = make_panel(100, 40, 2, 1, "structural", seed=11)
    affected = targets.columns[targets.isna().any()]

    assert len(affected) == round(0.15 * targets.shape[1])
    for column in affected:
        valid_positions = np.flatnonzero(targets[column].notna().to_numpy())
        np.testing.assert_array_equal(
            valid_positions,
            np.arange(valid_positions[0], valid_positions[-1] + 1),
        )


def test_tiny_case_produces_well_formed_record():
    record = run_case(
        "tiny",
        "clean",
        BenchmarkConfig(lambda_=0.0, ewma_halflife=None),
        seed=12,
    )

    assert record["dimensions"] == {
        "T": 500,
        "N": 20,
        "K": 3,
        "q": 2,
        "slow": False,
    }
    assert record["output_size_bytes"] > 0
    for phase in ("fit", "transform"):
        measurement = record["measurements"][phase]
        assert measurement["wall_seconds"] >= 0
        assert measurement["python_peak_bytes"] >= 0
        assert measurement["peak_rss_bytes"] > 0
        assert measurement["rss_increase_bytes"] >= 0
    assert set(record["measurements"]["accessors"]) == {
        "get_beta",
        "get_r2",
        "get_residuals",
        "get_se",
    }


def test_medium_tripwire_controls_exit_status(capsys):
    config = {"lambda_": 0.0, "ewma_halflife": None, "mode": "batched"}
    dimensions = {**SIZE_GRID["medium"].__dict__}

    def record(total: float) -> dict:
        measurement = {
            "wall_seconds": total / 2,
            "python_peak_bytes": 0,
            "peak_rss_bytes": 1,
            "rss_increase_bytes": 0,
        }
        return {
            "size": "medium",
            "dimensions": dimensions,
            "nan_pattern": "clean",
            "config": config,
            "measurements": {
                "fit": measurement,
                "transform": measurement,
                "accessors": {
                    name: None for name in ("get_beta", "get_r2", "get_residuals", "get_se")
                },
                "total_wall_seconds": total,
            },
            "output_size_bytes": 0,
        }

    baseline = {"records": [record(1.0)]}
    assert compare_payloads({"records": [record(3.0)]}, baseline)
    assert not compare_payloads({"records": [record(3.01)]}, baseline)
    capsys.readouterr()
