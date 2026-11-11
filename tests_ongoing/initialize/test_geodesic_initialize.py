from __future__ import annotations

from pathlib import Path

from popcornn.tools.tests import popcornn_run_test


def test_geodesic_initialize():
    popcornn_run_test(
        name="geodesic",
        config_path=Path("configs") / "initialize_geodesic.yaml",
        benchmark_path=Path("initialize") / "benchmarks"
    )
