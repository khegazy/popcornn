"""Per-stage convergence flags returned by ``Popcornn.optimize_path``.

Each stage contributes one bool: True if the ``threshold`` trigger fired,
False if the stage exhausted ``num_optimizer_iterations`` (always False
when ``threshold`` is None). ``max_batch`` is pinned so padaquad skips its
memory benchmark.
"""

from __future__ import annotations

import logging
import re

import pytest
import torch

from popcornn import Popcornn


def _stage(num_optimizer_iterations, threshold=None):
    return {
        "potential_params": {"name": "muller_brown"},
        "integrator_params": {
            "path_integrand_names": "pvre",
            "rtol": 1e-2,
            "atol": 1e-2,
            "max_batch": 100,
        },
        "optimizer_params": {
            "optimizer": {"name": "adam", "lr": 1e-3},
            "threshold": threshold,
            "patience": 1,
        },
        "num_optimizer_iterations": num_optimizer_iterations,
    }


def _run(**popcornn_kwargs):
    """Build the 2-image Muller-Brown Popcornn and run the three-stage chain.

    Stage 1 converges on its first iteration, stage 2 hits its iteration
    cap, stage 3 has no trigger: the progress table has 3 stage headers,
    3 column headers, 5 rows, 1 "converged" line, and 3 "stage done" lines.
    """
    torch.manual_seed(0)
    mep = Popcornn(
        images=[[-0.558, 1.442], [0.623, 0.028]],
        path_params={"name": "mlp", "width": 8, "depth": 2, "activation": "gelu"},
        device="cpu",
        dtype="float64",
        **popcornn_kwargs,
    )
    assert mep.converged is None

    converged = mep.optimize_path(
        _stage(3, threshold=1e10),  # trigger fires on the first iteration
        _stage(2, threshold=1e-12),  # unreachable trigger: iteration cap hit
        _stage(2, threshold=None),  # trigger disabled
    )
    return mep, converged


def test_optimize_path_returns_per_stage_converged_flags():
    mep, converged = _run()

    assert converged == [True, False, False]
    assert all(type(flag) is bool for flag in converged)
    assert mep.converged == converged


@pytest.mark.parametrize(
    ("quiet", "use_file"), [(False, False), (False, True), (True, True), (True, False)]
)
def test_progress_log_outputs(quiet, use_file, tmp_path, capsys, caplog):
    """The table goes once to each requested output and never through the root logger."""
    log_file = (
        tmp_path / "logs" / "run.log" if use_file else None
    )  # "logs" does not exist yet
    routed = (
        not quiet or use_file
    )  # otherwise popcornn attaches nothing and standard logging decides
    if not routed:
        caplog.set_level(logging.INFO, logger="popcornn")
    _run(quiet=quiet, log_file=log_file)

    screen = capsys.readouterr().out.splitlines()
    records = [r.getMessage() for r in caplog.records if r.name.startswith("popcornn")]
    file_lines = log_file.read_text(encoding="utf-8").splitlines() if use_file else []
    assert len(screen) == (0 if quiet else 15)
    assert len(file_lines) == (15 if use_file else 0)
    if screen and file_lines:
        assert (
            file_lines[0] == screen[0]
        )  # same stage header (rows differ only in timings)
    # caplog's handler is on the root logger: while popcornn routes the output, nothing may
    # reach it (a root handler, such as the one fairchem attaches, would print a second copy)
    assert len(records) == (0 if routed else 15)

    table = screen or file_lines or records
    assert sum(line.startswith("converged at iter 0") for line in table) == 1
    assert sum(line.startswith("stage ") and " done " in line for line in table) == 3

    package_logger = logging.getLogger("popcornn")
    assert package_logger.handlers == []
    assert package_logger.propagate is True
    if routed:
        assert package_logger.level == logging.NOTSET


def test_log_file_fixed_at_construction(tmp_path, monkeypatch):
    """A relative log_file names one file even if the working directory changes later."""
    monkeypatch.chdir(tmp_path)
    mep = Popcornn(
        images=[[-0.558, 1.442], [0.623, 0.028]],
        path_params={"name": "mlp", "width": 8, "depth": 2, "activation": "gelu"},
        device="cpu",
        dtype="float64",
        log_file="logs/run.log",
        quiet=True,
    )
    monkeypatch.chdir(tmp_path / "logs")  # has no "logs" subfolder
    mep.optimize_path(_stage(1))
    # stage header, column header, 1 row, stage-done line
    assert (
        len((tmp_path / "logs" / "run.log").read_text(encoding="utf-8").splitlines())
        == 4
    )


def test_init_logs_path_fit(capsys):
    """The constructor shows the path-fit messages, and nothing else, for intermediate images."""
    torch.manual_seed(0)
    Popcornn(
        images=[[-0.558, 1.442], [0.0, 0.8], [0.623, 0.028]],
        path_params={"name": "mlp", "width": 8, "depth": 2, "activation": "gelu"},
        device="cpu",
        dtype="float64",
    )
    screen = capsys.readouterr().out.splitlines()
    assert screen[0] == "Beginning path initialization"
    assert re.fullmatch(r"Finished path initialization after \d+ iterations", screen[1])
    assert len(screen) == 2
