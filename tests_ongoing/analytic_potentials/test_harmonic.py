from __future__ import annotations

from pathlib import Path

from popcornn.tools.tests import popcornn_run_test


def test_harmonic():
    popcornn_run_test(
        "harmonic",
        Path("configs") / "harmonic.yaml",
        Path("analytic_potentials") / "benchmarks"
    )
