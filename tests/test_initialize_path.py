from __future__ import annotations

import pytest
import torch

from popcornn.paths import get_path
from popcornn.tools import process_images


# TODO: Implement the test for initialize_path
@pytest.mark.skip(reason="initialize_path is not implemented yet")
def test_initialize_path():
    images = process_images("images/wolfe.json", device=torch.device("cpu"), dtype=dtype)
    path = get_path("mlp", images=images, unwrap_positions=False, device=torch.device("cpu"), dtype=dtype)
