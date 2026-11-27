"""Shared pytest configuration for popcornn's tests.

Holds what every test module relies on, so it is defined once:

- test ids that name data files by file name (``pytest_make_parametrize_id``);
- the same random state at the start of every test (``_seed_rng``);
- a pinned ``max_batch`` in every ``PathIntegrator``
  (``_forbid_unpinned_max_batch``).
"""

from __future__ import annotations

import functools
import inspect
from pathlib import Path

import pytest
import torch

from popcornn.tools import PathIntegrator
from tests import IMAGES_DIR, TEST_SEED, starting_mesh

_ORIGINAL_INIT = PathIntegrator.__init__
_INIT_SIGNATURE = inspect.signature(_ORIGINAL_INIT)


@functools.wraps(_ORIGINAL_INIT)
def _refuse_unpinned(self, *args, **kwargs):
    """``PathIntegrator.__init__`` that refuses ``max_batch=None``."""
    # Bind so max_batch is found whether it was passed by position or by name.
    if _INIT_SIGNATURE.bind(self, *args, **kwargs).arguments.get("max_batch") is None:
        msg = (
            "PathIntegrator was built with max_batch=None, so padaquad would search "
            "for a batch size by growing it until it runs out of memory. Pin "
            "max_batch, or mark a test of that search with "
            "@pytest.mark.unpinned_max_batch."
        )
        raise RuntimeError(msg)
    _ORIGINAL_INIT(self, *args, **kwargs)


@pytest.fixture(autouse=True, scope="session")
def _forbid_unpinned_max_batch():
    """Refuse an unpinned ``max_batch`` everywhere in the session.

    Without ``max_batch``, padaquad grows the batch 10x per pass until it runs
    out of memory, and popcornn's per-point gradient needs memory that grows
    like the square of the batch (#89). The batch it settles on depends on the
    machine, and a different batch changes results at the rounding level.
    Session scope also covers module- and session-scoped fixtures.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(PathIntegrator, "__init__", _refuse_unpinned)
        yield


@pytest.fixture(autouse=True)
def _allow_unpinned_max_batch(request, monkeypatch):
    """Lift the guard for tests marked ``unpinned_max_batch``; ``monkeypatch``
    puts it back after the test."""
    if request.node.get_closest_marker("unpinned_max_batch"):
        monkeypatch.setattr(PathIntegrator, "__init__", _ORIGINAL_INIT)


@pytest.fixture(autouse=True)
def _seed_rng():
    """Seed torch's global generator before every test.

    A test's starting point must not depend on which tests ran before it: an
    unseeded test that builds an MLP, or lets padaquad draw a random starting
    mesh, would otherwise start from whatever state earlier tests left.
    ``torch.manual_seed`` seeds the CPU generator and every CUDA device. Tests
    that need a different state reseed explicitly.
    """
    torch.manual_seed(TEST_SEED)


@pytest.fixture(
    params=[
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is not available."
            ),
        ),
    ],
    ids=["cpu", "cuda"],
)
def device(request):
    """Every device a test runs on: the CPU, and CUDA when it is available.
    A test with its own ``parametrize("device", ...)`` overrides this."""
    return request.param


@pytest.fixture(params=[torch.float32, torch.float64], ids=["float32", "float64"])
def dtype(request):
    """Every dtype a test runs in. A test's own ``parametrize`` overrides this."""
    return request.param


@pytest.fixture
def cheap_integrator_params():
    """``PathIntegrator`` settings for tests that check wiring, not accuracy.

    A new dict on every call, so no test can change another test's settings.
    Extend it by merging into a new dict,
    ``PathIntegrator(**{**cheap_integrator_params, "path_integrand_names": "pvre"})``;
    passing it next to a keyword it already holds raises ``TypeError``.

    - ``rtol=atol=1e-2``: explicit, so these tests do not depend on
      ``PathIntegrator``'s defaults, and coarse, which is enough for tests that
      check shapes, flags, finite values, log lines, call arguments and batch
      sizes (plan 00, Task 5a). Far too loose for numerical checks. Users:
      the ``track_ts`` tests (sample alignment), ``test_ts_end_to_end``
      (``ts_time`` in [0, 1], finite) and ``test_converged_flag`` (flags with
      thresholds 1e10 and 1e-12, far from any gradient norm; log lines).
    - ``max_batch=500``: pinned, so padaquad does not search for a batch size
      (the conftest guard refuses ``None``). It holds every node of these tests'
      integrals, so results are bit-identical to padaquad's own choice; it is
      above one gk21 panel (21 points), and it satisfies the memory rule
      ``max_batch**2 * width <= 3.2e7`` up to width 128.
    - ``runtime_checks=True``: padaquad raises on NaN or Inf. popcornn's default
      is False, which would let a NaN reach the gradient silently.

    - ``initial_mesh=starting_mesh(TEST_SEED)``: a fixed starting mesh, so
      padaquad draws no random mesh and results do not depend on how it
      draws one.

    ``integrate_gradient`` is left out: a test that is not about the gradient
    mode should add ``"integrate_gradient": False``, which is cheaper (#89).
    """
    return {
        "rtol": 1e-2,
        "atol": 1e-2,
        "max_batch": 500,
        "runtime_checks": True,
        "initial_mesh": starting_mesh(TEST_SEED),
    }


def pytest_make_parametrize_id(config, val, argname) -> str | None:
    """Name a test-data path by its file name in test ids.

    Paths are absolute (``IMAGES_DIR``), so without this the ids would contain
    the checkout's location and differ between machines.
    """
    if isinstance(val, str) and val.startswith(str(IMAGES_DIR)):
        return Path(val).name
    return None
