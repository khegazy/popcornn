"""Tests of the shared test machinery in ``tests/conftest.py``."""

from __future__ import annotations

import pytest
import torch

from tests import TEST_SEED


@pytest.mark.parametrize("repeat", [1, 2])
def test_each_test_starts_from_the_same_random_state(repeat):
    """Every test starts from torch.manual_seed(TEST_SEED), whatever ran before.
    The second case fails without the conftest fixture, because the first case
    has already advanced the global generator."""
    expected = torch.rand(4, generator=torch.Generator().manual_seed(TEST_SEED))
    assert torch.equal(torch.rand(4), expected)
