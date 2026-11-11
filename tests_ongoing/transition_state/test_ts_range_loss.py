from __future__ import annotations

from pathlib import Path

from popcornn.tools.tests import popcornn_run_test


def test_ts_range_loss():
    benchmark_path = Path("transition_state") / "benchmarks"
    for potential in ["wolfe_schlegel", "morse"]:
        # Test high level functionality
        name = f"ts_range_loss_{potential}"
        config_path = Path("configs") / f"{name}.yaml"
        popcornn_run_test(name, config_path, benchmark_path)
