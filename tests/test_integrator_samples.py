"""Tests for ``PathIntegrator(track_ts=True)`` on the padaquad backend.

Exercises the tracked-variables path: the integrand emits per-quadrature-point
``(energies, dE/dt)`` alongside the gradient, padaquad returns them at the
accepted nodes, and ``PathIntegrator`` reassembles them into a ``SamplesCache``
so the transition-state finder consumes the integrator's own samples without
paying for any extra path forwards.
"""
from __future__ import annotations

import pytest
import torch

from popcornn.paths import get_path
from popcornn.potentials import get_potential
from popcornn.tools import PathIntegrator, SamplesCache, process_images
from popcornn.tools.integrand import evaluate_integrand_sum


@pytest.fixture
def muller_brown_setup():
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float64
    images = process_images(
        "tests/images/muller_brown.json", device=device, dtype=dtype
    )
    path = get_path("mlp", images=images, device=device, dtype=dtype)
    potential = get_potential(
        "muller_brown", images=images, device=device, dtype=dtype,
    )
    path.set_potential(potential)
    return path, device, dtype


def test_track_ts_aligned_with_quadrature_mesh(muller_brown_setup):
    path, device, dtype = muller_brown_setup
    integrator = PathIntegrator(
        method="gk21",
        path_integrand_names="pvre",
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
        method="gk21",
        path_integrand_names="pvre",
        rtol=1e-2, atol=1e-2,
        track_ts=False,
        device=device, dtype=dtype,
    )

    out = integrator.integrate_path(path)
    assert out.samples is None


def test_batch_size_measured_once(muller_brown_setup, monkeypatch):
    """With ``max_batch=None``, padaquad measures the memory of one integrand
    evaluation (``_setup_memory_checks``) on the first ``integrate_path`` and
    stores the resulting batch size on the solver (``previous_max_batch``).
    ``PathIntegrator`` passes ``same_f=True``, so later calls reuse the stored
    size: the slow measurement runs once per integrator, and never on the loss
    solver, which starts from the gradient solver's size.
    """
    path, device, dtype = muller_brown_setup
    integrator = PathIntegrator(
        method="gk21",
        path_integrand_names="pvre",
        rtol=1e-2, atol=1e-2,
        track_loss=True,
        device=device, dtype=dtype,
    )
    n_measurements = {"main": 0, "loss": 0}
    for name, solver in (("main", integrator._solver), ("loss", integrator._loss_solver)):
        def counting(*args, _measure=solver._setup_memory_checks, _name=name, **kwargs):
            n_measurements[_name] += 1
            return _measure(*args, **kwargs)
        monkeypatch.setattr(solver, "_setup_memory_checks", counting)

    out = integrator.integrate_path(path)
    assert torch.isfinite(out.integral).all()
    measured = integrator._solver.previous_max_batch
    assert isinstance(measured, int) and measured > 0

    integrator.integrate_path(path)
    assert n_measurements == {"main": 1, "loss": 0}
    assert integrator._solver.previous_max_batch == measured


def _simulate_oom(monkeypatch, limit, integrand):
    """Make one path integrand raise ``torch.OutOfMemoryError`` on batches of
    more than ``limit`` points, and record that integrand's batch sizes.

    ``integrand`` selects which one: ``'main'`` (the ``_solver`` integrand,
    which passes ``also_resolve``) or ``'loss'`` (the detached ``track_loss``
    integrand, which does not). Returns the list the batch sizes are appended to.
    """
    batch_sizes = []

    def evaluate_with_limit(terms, t, path, **kwargs):
        if ("also_resolve" in kwargs) == (integrand == "main"):
            batch_sizes.append(t.shape[0])
            if t.shape[0] > limit:
                raise torch.OutOfMemoryError(f"simulated out-of-memory error at {t.shape[0]} points")
        return evaluate_integrand_sum(terms, t, path, **kwargs)

    monkeypatch.setattr(
        "popcornn.tools.integrator.evaluate_integrand_sum", evaluate_with_limit
    )
    return batch_sizes


@pytest.mark.parametrize("integrand", ["main", "loss"])
def test_oom_batch_reduction_persists(muller_brown_setup, monkeypatch, integrand):
    """An out-of-memory error makes padaquad shrink the batch and retry. With
    ``same_f=True`` each solver keeps its reduced size, so the error must not
    recur on the next ``integrate_path``. ``'main'`` exercises ``_solver``,
    ``'loss'`` exercises ``_loss_solver`` on its own.
    """
    path, device, dtype = muller_brown_setup
    limit = 50  # more than gk21's 23 nodes per panel, so one panel always fits
    batch_sizes = _simulate_oom(monkeypatch, limit, integrand)
    integrator = PathIntegrator(
        method="gk21",
        path_integrand_names="pvre",
        rtol=1e-2, atol=1e-2,
        max_batch=400,
        track_loss=True,
        device=device, dtype=dtype,
    )

    integrator.integrate_path(path)
    assert max(batch_sizes) > limit  # the first call hit the simulated error

    batch_sizes.clear()
    integrator.integrate_path(path)
    assert max(batch_sizes) <= limit


def test_oom_raises_without_integrate_gradient(muller_brown_setup, monkeypatch):
    """With ``integrate_gradient=False`` padaquad backpropagates through each
    batch, so it cannot shrink the batch and retry: the error reaches the caller.
    """
    path, device, dtype = muller_brown_setup
    _simulate_oom(monkeypatch, 50, "main")
    integrator = PathIntegrator(
        method="gk21",
        path_integrand_names="pvre",
        rtol=1e-2, atol=1e-2,
        max_batch=400,
        integrate_gradient=False,
        device=device, dtype=dtype,
    )
    with pytest.raises(torch.OutOfMemoryError):
        integrator.integrate_path(path)


def test_gpu_index_reaches_padaquad(monkeypatch):
    """The GPU index of ``device`` must reach padaquad, so a ``cuda:1`` run
    integrates on GPU 1. Runs without a GPU: only the solver construction is
    intercepted.
    """
    devices = []

    def record_device(*args, **kwargs):
        devices.append(kwargs["device"])
        return object()

    monkeypatch.setattr("popcornn.tools.integrator.adaptive_quadrature", record_device)
    PathIntegrator(
        path_integrand_names="pvre",
        device=torch.device("cuda", 1), dtype=torch.float64,
    )
    assert devices == [torch.device("cuda", 1)] * 2
