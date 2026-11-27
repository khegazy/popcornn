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

    It is exactly the mesh padaquad 2.0 draws for itself after
    ``torch.manual_seed(seed)`` (``_setup_initial_mesh`` with its default
    ``N_init_steps=13``; ``tests/test_conftest.py`` checks this), but drawn from
    a local generator, so torch's global generator is never touched: 3 equal
    segments, each with 4 sorted random points; the first segment is shifted
    to start at 0 and the last to end at 1, giving 12 points (11 subintervals).
    Irregular on purpose: an evenly spaced mesh can put a corner of the
    integrand (for example t = 0.5 on the sphere path) on a panel boundary and
    integrate it exactly, which hides quadrature errors.

    Parameters
    ----------
    seed : int
        Seed of the local generator; each seed gives a different mesh.

    Returns
    -------
    torch.Tensor
        Strictly increasing times, shape [12, 1], float64, on the CPU.
    """
    generator = torch.Generator().manual_seed(seed)
    n_segments, points_per_segment = 3, 4
    lo = torch.zeros(1, dtype=torch.float64)
    hi = torch.ones(1, dtype=torch.float64)
    # Same operations, in the same order, as padaquad's fresh mesh, so the
    # result is bit-identical (padaquad draws float32 numbers with torch.rand).
    dt = (hi - lo) / n_segments
    mesh = lo + dt * torch.arange(n_segments)[:, None, None]
    offsets = dt * torch.rand((n_segments, points_per_segment, 1), generator=generator)
    mesh = mesh + torch.sort(offsets, dim=1)[0]
    mesh[0] += lo - mesh[0, 0]
    mesh[-1] += hi - mesh[-1, -1]
    mesh = torch.flatten(mesh, start_dim=0, end_dim=1)
    mesh[0] = lo
    mesh[-1] = hi
    if not torch.all(mesh[1:] > mesh[:-1]):
        msg = f"starting_mesh({seed}) is not strictly increasing; choose another seed."
        raise ValueError(msg)
    return mesh
