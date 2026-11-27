from __future__ import annotations

import pytest
import torch

from popcornn.paths import get_path
from popcornn.tools import process_images
from tests import IMAGES_DIR

# TODO: Implement the test for initialize_path
dtype = torch.float64


@pytest.mark.skip(reason="initialize_path is not implemented yet")
def test_initialize_path():
    images = process_images(
        str(IMAGES_DIR / "wolfe.json"), device=torch.device("cpu"), dtype=dtype
    )
    _path = get_path(
        "mlp",
        images=images,
        unwrap_positions=False,
        device=torch.device("cpu"),
        dtype=dtype,
    )
