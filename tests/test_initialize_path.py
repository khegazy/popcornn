"""Tests for ``initialize_path``: fitting a path through intermediate images.

Plan: ``tests_ongoing/test_plans/03_initialize_path_implementation_plan.md``.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch
from ase import Atoms

from popcornn import Popcornn
from popcornn.optimization import initialize_path
from popcornn.optimization.initialize_path import INIT_LOSS_THRESHOLD
from popcornn.paths import get_path
from popcornn.tools import process_images
from tests import IMAGES_DIR

# Müller-Brown reactant, one intermediate image, product.
MB_ALL = str(IMAGES_DIR / "muller_brown_all.json")


def _mb_setup(device, dtype):
    """Müller-Brown images, a fresh MLP path at the default width, and evenly spaced times."""
    images = process_images(MB_ALL, device=device, dtype=dtype)
    path = get_path("mlp", images=images, device=device, dtype=dtype)
    times = torch.linspace(0, 1, len(images), device=device, dtype=dtype)[:, None]
    return images, path, times


def _assert_fits(path, times, targets, slack=0.0):
    """Assert the path meets ``initialize_path``'s stopping rule at ``times``.

    MSE < INIT_LOSS_THRESHOLD over all N images x D coordinates bounds every
    coordinate's error by sqrt(N * D * INIT_LOSS_THRESHOLD): 2.45e-4 for
    Müller-Brown (N=3, D=2), 3.0e-4 for the periodic test (N=3, D=3). ``slack``
    is added to the threshold for a caller whose MSE is not computed the way
    ``loss_init`` computes it.
    """
    with torch.no_grad():
        pred = path(times).positions
    mse = ((pred - targets) ** 2).mean().item()
    assert mse < INIT_LOSS_THRESHOLD + slack, f"MSE {mse:.3e}"


def test_initialize_path_fits_images(device, dtype, caplog):
    """The fitted path passes through all three images, and a converged fit does not warn."""
    images, path, times = _mb_setup(device, dtype)

    with caplog.at_level(logging.WARNING, logger="popcornn"):
        assert initialize_path(path, times, images.positions) is path

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    _assert_fits(path, times, images.positions)


def test_popcornn_fits_intermediate_images():
    """``Popcornn`` fits its path through all images at evenly spaced times.

    CPU and float32 (``Popcornn``'s default) only: this checks the wiring, which
    does not depend on device or dtype. The times are one-dimensional, as
    ``Popcornn.__init__`` passes them.
    """
    mep = Popcornn(images=MB_ALL, path_params={"name": "mlp"}, device="cpu")
    times = torch.linspace(0, 1, len(mep.images), device=mep.device, dtype=mep.dtype)
    _assert_fits(mep.path, times, mep.images.positions)


def test_initialize_path_periodic(device, dtype):
    """An image stored wrapped across the cell is fitted to its unwrapped position.

    One H atom in a 10 Å cubic cell. ``process_images`` unwraps the middle image
    against the reactant to x = -0.5, outside the cell, while the path's output is
    always wrapped into the cell (x = 9.5). ``loss_init`` must take the
    minimum-image displacement, so that the fitted path crosses the boundary the
    short way (unwrapped x = -0.5). A path that runs back across the cell
    (unwrapped x = 9.5) has the same wrapped output, so the unwrapped position is
    checked as well as the wrapped one.
    """
    stored = np.array([[1.0, 5.0, 5.0], [9.5, 6.0, 5.0], [2.0, 5.0, 5.0]])
    frames = [
        Atoms("H", positions=[p], cell=10.0 * np.eye(3), pbc=True) for p in stored
    ]
    # process_images unwraps the frames in place, so the reference below uses `stored`.
    images = process_images(frames, device=device, dtype=dtype)
    assert images.positions[1, 0] < 0  # only the middle image is outside the cell
    path = get_path("mlp", images=images, device=device, dtype=dtype)
    times = torch.linspace(0, 1, len(images), device=device, dtype=dtype)[:, None]

    initialize_path(path, times, images.positions)

    # The wrapped path output can be compared with the stored, wrapped positions
    # directly: every stored coordinate is at least 0.5 Å from a cell face, and the
    # fit error is about 3e-4 Å, so no coordinate wraps differently. ``loss_init``
    # wraps the displacement instead, so in float32 the two MSEs differ by rounding
    # at the 10 Å scale: at most 4.8e-11 over 30 seeds on CPU and 3.3e-11 on CUDA.
    # The slack is 10x that; it is far below the MSE of a wrong fit (> 1e-5).
    _assert_fits(
        path, times, torch.tensor(stored, device=device, dtype=dtype), slack=5e-10
    )
    # Together with the wrapped fit above, this pins the unwrapped middle image near
    # x = -0.5 (short way), not 9.5 (long way round the cell).
    with torch.no_grad():
        assert path.get_positions(times)[1, 0] < 0


def test_initialize_path_keeps_converged_path(caplog):
    """A path that already meets the fit threshold is returned unchanged, without a WARNING.

    The targets sit 1e-4 off the path at the middle image, so the starting MSE is
    2 * (1e-4)**2 / 6 = 3.3e-9, below the threshold, while the loss gradient is not
    zero. A fit that takes an Adam step after its check passes moves the parameters
    (measured MSE afterwards: 2.4e-5); the default ``max_steps`` call catches that.
    The ``max_steps=0`` call catches a fit that warns whenever it reaches the cap,
    even when the path meets the threshold there.
    """
    _, path, times = _mb_setup(torch.device("cpu"), torch.float64)
    with torch.no_grad():
        targets = path(times).positions.clone()
    targets[1] += 1e-4
    before = [p.detach().clone() for p in path.parameters()]

    with caplog.at_level(logging.WARNING, logger="popcornn"):
        initialize_path(path, times, targets)
        initialize_path(path, times, targets, max_steps=0)

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert all(
        torch.equal(old, new)
        for old, new in zip(before, path.parameters(), strict=True)
    )


@pytest.mark.parametrize("max_steps", [0, 5])
def test_initialize_path_warns_at_max_steps(caplog, max_steps):
    """A fit stopped by ``max_steps`` logs one WARNING with the step count (#91).

    After at most 5 Adam steps at lr = 1e-3 the loss is about 0.27, far above the
    threshold. ``initialize_path`` is called directly: inside ``Popcornn.__init__``
    the ``popcornn`` logger does not propagate to caplog's root handler.
    """
    images, path, times = _mb_setup(torch.device("cpu"), torch.float64)

    with caplog.at_level(logging.WARNING, logger="popcornn"):
        initialize_path(path, times, images.positions, max_steps=max_steps)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert f"did not converge after {max_steps} iterations" in warnings[0].getMessage()


def test_initialize_path_rejects_negative_max_steps():
    images, path, times = _mb_setup(torch.device("cpu"), torch.float64)
    with pytest.raises(ValueError, match="max_steps"):
        initialize_path(path, times, images.positions, max_steps=-1)
