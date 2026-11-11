from __future__ import annotations

from pathlib import Path

from popcornn.tools.tests import popcornn_run_test


def test_wolfe_schlegel():
    popcornn_run_test(
        name="wolfe_schlegel",
        config_path=Path("configs") / "wolfe_schlegel.yaml",
        benchmark_path=Path("analytic_potentials") / "benchmarks"
    )
