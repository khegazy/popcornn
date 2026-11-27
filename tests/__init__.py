"""popcornn test suite: shared constants and helpers for the test modules."""

from __future__ import annotations

from pathlib import Path

import torch

# Test data, found relative to this package so tests run from any directory.
IMAGES_DIR = Path(__file__).parent / "images"

# Seed of torch's global generator at the start of every test (conftest.py).
TEST_SEED = 0


def starting_mesh(seed: int) -> torch.Tensor:
    """A fixed, irregular starting mesh on [0, 1] for ``PathIntegrator(initial_mesh=...)``.

    Without one, padaquad draws a random starting mesh from torch's global
    generator, so results would follow padaquad's way of drawing it. This mesh
    has the layout of padaquad's default fresh mesh (``N_init_steps=13``): 3
    equal segments with 4 sorted random points each, then the first point set
    to 0 and a last point at 1, 13 points in all. The points come from a local
    generator, so the global one is never touched. Irregular on purpose: an
    evenly spaced mesh can put a corner of the integrand (for example t = 0.5 on
    the sphere path) on a panel boundary and integrate it exactly, which hides
    quadrature errors.

    Parameters
    ----------
    seed : int
        Seed of the local generator; each seed gives a different mesh.

    Returns
    -------
    torch.Tensor
        Strictly increasing times, shape [13, 1], float64, on the CPU.
    """
    generator = torch.Generator().manual_seed(seed)
    n_segments, points_per_segment = 3, 4
    offsets = torch.rand(
        n_segments, points_per_segment, generator=generator, dtype=torch.float64
    )
    starts = torch.arange(n_segments, dtype=torch.float64)[:, None]
    points = ((starts + offsets.sort(dim=1).values) / n_segments).flatten()
    points[0] = 0.0
    mesh = torch.cat([points, torch.ones(1, dtype=torch.float64)])[:, None]
    if not torch.all(mesh[1:] > mesh[:-1]):
        msg = f"starting_mesh({seed}) is not strictly increasing; choose another seed."
        raise ValueError(msg)
    return mesh
