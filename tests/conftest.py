"""Shared pytest configuration for popcornn's tests.

Holds what every test module relies on, so it is defined once:

- test ids that name data files by file name (``pytest_make_parametrize_id``);
- the same random state at the start of every test (``_seed_rng``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests import IMAGES_DIR, TEST_SEED


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
