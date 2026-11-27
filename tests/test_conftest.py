"""Tests of the shared test machinery in ``tests/conftest.py``."""

from __future__ import annotations

import pytest
import torch

from popcornn.tools import PathIntegrator
from tests import TEST_SEED


@pytest.mark.parametrize("repeat", [1, 2])
def test_each_test_starts_from_the_same_random_state(repeat):
    """Every test starts from torch.manual_seed(TEST_SEED), whatever ran before.
    The second case fails without the conftest fixture, because the first case
    has already advanced the global generator."""
    expected = torch.rand(4, generator=torch.Generator().manual_seed(TEST_SEED))
    assert torch.equal(torch.rand(4), expected)


def test_unpinned_max_batch_is_refused():
    """Every test pins max_batch: without it padaquad searches for a batch size
    by growing it until it runs out of memory, which depends on the machine."""
    with pytest.raises(RuntimeError, match="unpinned_max_batch"):
        PathIntegrator(path_integrand_names="pvre", rtol=1e-2, atol=1e-2)


@pytest.mark.unpinned_max_batch
def test_unpinned_max_batch_allowed_when_marked():
    """A test that tests padaquad's batch-size search opts out with the marker."""
    PathIntegrator(path_integrand_names="pvre", rtol=1e-2, atol=1e-2)
