"""Tests for ``PathIntegrator.save_samples=True``.

Exercises the side-buffer + byte-keyed stitch that lets the
transition-state finder consume the integrator's quadrature samples
without paying for any extra path forwards.
"""

import pytest
import torch

from popcornn.paths import get_path
from popcornn.potentials import get_potential
from popcornn.tools import PathIntegrator, SamplesCache, process_images


@pytest.fixture
def muller_brown_setup():
    torch.manual_seed(0)
    device = torch.device('cpu')
    dtype = torch.float64
    images = process_images(
        'tests/images/muller_brown.json', device=device, dtype=dtype
    )
    path = get_path(
        'mlp', images=images, n_embed=4, depth=2,
        activation='gelu', device=device, dtype=dtype,
    )
    potential = get_potential(
        'muller_brown', images=images, device=device, dtype=dtype,
    )
    path.set_potential(potential)
    return path, device, dtype


def test_save_samples_aligned_with_quadrature_mesh(muller_brown_setup):
    path, device, dtype = muller_brown_setup
    integrator = PathIntegrator(
        method='gk21',
        path_integrand_names='pvre',
        rtol=1e-2, atol=1e-2,
        save_samples=True,
        device=device, dtype=dtype,
    )

    out = integrator.integrate_path(path)

    assert isinstance(out.samples, SamplesCache)
    expected_n = out.t.flatten().shape[0]
    assert out.samples.time.shape == (expected_n,)
    # dE/dt is a per-sample scalar.
    assert out.samples.dEdt.shape == (expected_n,)
    # energies may be shape [N, 1] or [N, K]; just assert leading axis.
    assert out.samples.energies.shape[0] == expected_n

    # sample times round-trip to the integrator's accepted mesh exactly.
    assert torch.allclose(
        out.samples.time, out.t.flatten().to(out.samples.time.device)
    )
    # nothing nan / inf — energies and dE/dt actually came from the potential.
    assert torch.isfinite(out.samples.energies).all()
    assert torch.isfinite(out.samples.dEdt).all()


def test_save_samples_off_yields_none(muller_brown_setup):
    path, device, dtype = muller_brown_setup
    integrator = PathIntegrator(
        method='gk21',
        path_integrand_names='pvre',
        rtol=1e-2, atol=1e-2,
        save_samples=False,
        device=device, dtype=dtype,
    )

    out = integrator.integrate_path(path)
    assert out.samples is None


def test_sticky_max_batch_after_oom_in_integrate_path(muller_brown_setup):
    """If torchpathint reports a shrunken max_batch, ``PathIntegrator``
    stores it so the next ``integrate_path`` starts at the learned size
    rather than re-discovering it from scratch.

    Simulated by monkeypatching ``path_integral`` (we can't provoke a real
    CUDA OOM in a unit test); the mechanics on top of torchpathint are
    what's under test here, not the shrinker itself.
    """
    path, device, dtype = muller_brown_setup
    integrator = PathIntegrator(
        method='gk21',
        path_integrand_names='pvre',
        rtol=1e-2, atol=1e-2,
        device=device, dtype=dtype,
    )

    # Confirm normal-path behavior leaves max_batch == None when no OOM.
    out = integrator.integrate_path(path)
    assert integrator.max_batch == out.max_batch  # both None after a clean run

    # Now simulate torchpathint reporting a learned, shrunken value.
    import popcornn.tools.integrator as integ_mod

    real_pi = integ_mod.path_integral

    def fake_pi(*args, **kwargs):
        result = real_pi(*args, **kwargs)
        result.max_batch = 7  # pretend the shrinker landed at 7
        return result

    integ_mod.path_integral = fake_pi
    try:
        integrator.integrate_path(path)
    finally:
        integ_mod.path_integral = real_pi
    assert integrator.max_batch == 7
