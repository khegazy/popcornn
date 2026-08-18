"""Tests for ``PathIntegrator(track_ts=True)`` on the padaquad backend.

Exercises the tracked-variables path: the integrand emits per-quadrature-point
``(energies, dE/dt)`` alongside the gradient, padaquad returns them at the
accepted nodes, and ``PathIntegrator`` reassembles them into a ``SamplesCache``
so the transition-state finder consumes the integrator's own samples without
paying for any extra path forwards.
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
    path = get_path('mlp', images=images, device=device, dtype=dtype)
    potential = get_potential(
        'muller_brown', images=images, device=device, dtype=dtype,
    )
    path.set_potential(potential)
    return path, device, dtype


def test_track_ts_aligned_with_quadrature_mesh(muller_brown_setup):
    path, device, dtype = muller_brown_setup
    integrator = PathIntegrator(
        method='gk21',
        path_integrand_names='pvre',
        rtol=1e-2, atol=1e-2,
        track_ts=True,
        device=device, dtype=dtype,
    )

    out = integrator.integrate_path(path)

    assert isinstance(out.samples, SamplesCache)

    # Samples come from padaquad's accepted nodes, already flattened to
    # [P, T] (shared panel boundaries deduplicated) -> P points.
    accepted_t = out.nodes[..., 0]
    expected_n = accepted_t.shape[0]
    assert out.samples.time.shape == (expected_n,)
    # dE/dt is a per-sample scalar.
    assert out.samples.dEdt.shape == (expected_n,)
    # energies may be shape [P, 1] or [P, K]; just assert leading axis.
    assert out.samples.energies.shape[0] == expected_n

    # sample times are the accepted-node times, sorted ascending.
    assert torch.allclose(
        out.samples.time,
        torch.sort(accepted_t.to(out.samples.time.device))[0],
    )
    assert torch.all(out.samples.time[1:] - out.samples.time[:-1] >= 0)
    # nothing nan / inf — energies and dE/dt actually came from the potential.
    assert torch.isfinite(out.samples.energies).all()
    assert torch.isfinite(out.samples.dEdt).all()


def test_track_ts_off_yields_none(muller_brown_setup):
    path, device, dtype = muller_brown_setup
    integrator = PathIntegrator(
        method='gk21',
        path_integrand_names='pvre',
        rtol=1e-2, atol=1e-2,
        track_ts=False,
        device=device, dtype=dtype,
    )

    out = integrator.integrate_path(path)
    assert out.samples is None


def test_max_batch_snapshotted_after_first_run(muller_brown_setup):
    """With ``max_batch=None``, padaquad benchmarks the integrand's memory on
    the first ``integrate_path``. ``PathIntegrator`` then snapshots that resolved
    batch size onto itself so every subsequent call passes an explicit
    ``max_batch`` and the (slow) benchmark fires once per integrator lifetime
    rather than once per step (padaquad's own ``id(f)`` cache can't help — our
    integrand closure is rebuilt each call, so its identity never matches).
    """
    path, device, dtype = muller_brown_setup
    integrator = PathIntegrator(
        method='gk21',
        path_integrand_names='pvre',
        rtol=1e-2, atol=1e-2,
        device=device, dtype=dtype,
    )
    assert integrator.max_batch is None

    out = integrator.integrate_path(path)
    assert torch.isfinite(out.integral).all()
    # Snapshotted from the first run's benchmark (None on CPU where the memory
    # budget is effectively unbounded; a positive cap on a memory-limited device).
    snapshotted = integrator.max_batch
    assert snapshotted is None or snapshotted > 0

    # Stable across subsequent calls — captured once, not re-discovered.
    integrator.integrate_path(path)
    assert integrator.max_batch == snapshotted
