from __future__ import annotations

from pathlib import Path

from popcornn.tools.tests import popcornn_run_test


def test_lennard_jones():
    popcornn_run_test(
        name="lennard_jones",
        config_path=Path("configs") / "lennard_jones.yaml",
        benchmark_path=Path("analytic_potentials") / "benchmarks"
    )
