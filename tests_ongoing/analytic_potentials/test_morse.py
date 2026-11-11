from __future__ import annotations

from pathlib import Path

from popcornn.tools.tests import popcornn_run_test


def test_morse():
    popcornn_run_test(
        name="morse",
        config_path=Path("configs") / "morse.yaml",
        benchmark_path=Path("analytic_potentials") / "benchmarks"
    )
