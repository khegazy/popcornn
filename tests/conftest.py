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
from tests import IMAGES_DIR, TEST_SEED

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


def pytest_make_parametrize_id(config, val, argname) -> str | None:
    """Name a test-data path by its file name in test ids.

    Paths are absolute (``IMAGES_DIR``), so without this the ids would contain
    the checkout's location and differ between machines.
    """
    if isinstance(val, str) and val.startswith(str(IMAGES_DIR)):
        return Path(val).name
    return None
