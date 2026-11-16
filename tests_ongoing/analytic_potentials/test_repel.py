from __future__ import annotations

from pathlib import Path

from popcornn.tools.tests import popcornn_run_test


def test_repel():
    popcornn_run_test(
        name="repel",
        config_path=Path("configs") / "repel.yaml",
        benchmark_path=Path("analytic_potentials") / "benchmarks"
    )
