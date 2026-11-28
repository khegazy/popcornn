"""popcornn test suite: shared constants and helpers for the test modules."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from popcornn.paths import get_path
from popcornn.potentials import get_potential
from popcornn.tools import process_images

if TYPE_CHECKING:
    from popcornn.paths.base_path import BasePath

# Test data, found relative to this package so tests run from any directory.
IMAGES_DIR = Path(__file__).parent / "images"

# PathIntegrator tolerances for tests that check wiring (shapes, flags, call
# arguments, batch sizes), not accuracy; far too loose for numerical checks.
# Used by conftest.cheap_integrator_params and by test-specific settings.
CHEAP_TOLERANCES = {"rtol": 1e-2, "atol": 1e-2}

# Seed of torch's global generator at the start of every test (conftest.py).
TEST_SEED = 0

# Limit for comparing a computation with an independent closed-form or explicit
# reference that differs from it only by rounding, relative to the reference's
# largest value. Used by test_potentials.py (pair potentials against a pair loop)
# and test_integrator_gradient.py (the sphere's exact pvre loss and gradient).
# Measured worst for the pair potentials, CPU and A100 (2026-11-14): float64
# 1.7e-15, float32 1.4e-6. The float32 value assumes coordinates within a few
# Angstrom of the origin.
REFERENCE_TOL = {torch.float64: 1e-10, torch.float32: 2e-5}


def make_path(
    images: str,
    potential: str,
    device: torch.device,
    dtype: torch.dtype,
    name: str = "mlp",
    **path_kwargs,
) -> BasePath:
    """A path between the images in ``images``, with its potential attached.

    The same three steps as ``Popcornn`` (``process_images``, ``get_path``,
    ``set_potential``), without the optimizer around them. The path's weights
    come from torch's global generator, which the conftest seeds per test.

    Parameters
    ----------
    images : str
        Image file, normally ``str(IMAGES_DIR / "<file>")``.
    potential : str
        Potential name for ``get_potential``.
    device, dtype
        Where and in what precision to build everything.
    name : str, default "mlp"
        Path name for ``get_path``.
    **path_kwargs
        Further ``get_path`` arguments, for example ``width``.

    Returns
    -------
    BasePath
        The path, with ``set_potential`` already called.
    """
    processed = process_images(images, device=device, dtype=dtype)
    path = get_path(name, images=processed, device=device, dtype=dtype, **path_kwargs)
    path.set_potential(
        get_potential(potential, images=processed, device=device, dtype=dtype)
    )
    return path


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
