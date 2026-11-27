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
from tests import IMAGES_DIR


@pytest.fixture
def muller_brown_setup():
    device = torch.device("cpu")
    dtype = torch.float64
    images = process_images(
        str(IMAGES_DIR / "muller_brown.json"),
        device=device,
        dtype=dtype,
    )
    path = get_path("mlp", images=images, device=device, dtype=dtype)
    potential = get_potential(
        "muller_brown",
        images=images,
        device=device,
        dtype=dtype,
    )
    path.set_potential(potential)
    return path, device, dtype


def test_track_ts_aligned_with_quadrature_mesh(muller_brown_setup):
    path, device, dtype = muller_brown_setup
    integrator = PathIntegrator(
        method="gk21",
        path_integrand_names="pvre",
        rtol=1e-2,
        atol=1e-2,
        max_batch=500,  # holds every node: results equal padaquad's own batch size
        track_ts=True,
        device=device,
        dtype=dtype,
    )

    out = integrator.integrate_path(path)

    assert isinstance(out.samples, SamplesCache)

    # Samples come from padaquad's accepted nodes, already flattened to
    # [P, T] -> P points.
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
        rtol=1e-2,
        atol=1e-2,
        max_batch=500,  # holds every node: results equal padaquad's own batch size
        track_ts=False,
        device=device,
        dtype=dtype,
    )

    out = integrator.integrate_path(path)
    assert out.samples is None


def _record_integrate_calls(monkeypatch, integrator):
    """Wrap both padaquad solvers' ``integrate`` to record, per call, the
    ``state`` passed and the state returned. Returns
    ``{"main": [...], "loss": [...]}`` of ``(state_in, state_out)``."""
    calls = {"main": [], "loss": []}
    for name, solver in (
        ("main", integrator._solver),
        ("loss", integrator._loss_solver),
    ):

        def recording(*args, _integrate=solver.integrate, _name=name, **kwargs):
            result, state = _integrate(*args, **kwargs)
            calls[_name].append((kwargs["state"], state))
            return result, state

        monkeypatch.setattr(solver, "integrate", recording)
    return calls


# Tolerances and batch size shared by the warm-start tests: loose tolerances
# keep them fast; max_batch=400 holds every batch of these integrals and skips
# padaquad's batch-size search.
_WARM_START_PARAMS = {
    "method": "gk21",
    "path_integrand_names": "pvre",
    "rtol": 1e-2,
    "atol": 1e-2,
    "max_batch": 400,
    "track_loss": True,
}


@pytest.mark.parametrize(
    "max_batch",
    [
        pytest.param(400, id="pinned"),
        pytest.param(None, id="found", marks=pytest.mark.unpinned_max_batch),
    ],
)
def test_warm_start_passes_previous_state(muller_brown_setup, monkeypatch, max_batch):
    """Each call starts from the state its own solver returned last (mesh and
    batch size); the first call of each solver starts from none, so the loss
    solver finds its own batch size rather than the gradient solver's.
    ``max_batch=None`` (popcornn's default, the one exception to pinning it in
    tests) lets padaquad find the batch size on the first call; that it then
    reuses the state's size instead of searching again is padaquad's to test.
    """
    path, device, dtype = muller_brown_setup
    params = _WARM_START_PARAMS | {"max_batch": max_batch}
    integrator = PathIntegrator(**params, device=device, dtype=dtype)
    calls = _record_integrate_calls(monkeypatch, integrator)

    out = integrator.integrate_path(path)
    integrator.integrate_path(path)

    assert torch.isfinite(out.integral).all()
    for name in ("main", "loss"):
        (state_in_1, state_out_1), (state_in_2, _) = calls[name]
        assert state_in_1 is None
        assert state_in_2 is state_out_1
        if max_batch is None:
            assert state_out_1.max_batch > 0
        else:
            assert state_out_1.max_batch == max_batch
    assert integrator._state is calls["main"][1][1]
    assert integrator._loss_state is calls["loss"][1][1]


@pytest.mark.parametrize(
    ("first_final", "second_final"),
    [(0.5, 1.0), (1.0, 0.5)],
    ids=["extend", "cut"],
)
def test_bounds_relative_to_previous_mesh(
    muller_brown_setup, monkeypatch, first_final, second_final
):
    """A call whose bounds differ from the previous mesh's ends still starts
    from that mesh: padaquad extends it to a bound outside it and cuts it at a
    bound inside it. The result must match a fresh integration of the second
    interval to within the quadrature tolerance."""
    path, device, dtype = muller_brown_setup
    integrator = PathIntegrator(**_WARM_START_PARAMS, device=device, dtype=dtype)
    calls = _record_integrate_calls(monkeypatch, integrator)

    torch.manual_seed(0)
    integrator.integrate_path(path, t_final=first_final)
    out = integrator.integrate_path(path, t_final=second_final)

    (_, first_state), (state_2, second_state) = calls["main"]
    assert state_2 is first_state
    assert second_state.mesh[-1].item() == second_final

    torch.manual_seed(0)
    fresh = PathIntegrator(**_WARM_START_PARAMS, device=device, dtype=dtype)
    expected = fresh.integrate_path(path, t_final=second_final)
    # padaquad bounds each panel's estimated error in the L2 norm (norm="2") by
    # max(atol, rtol*||I||) = 1e-2*||I|| here (||I|| ~ 80-150), so compare the
    # whole vector in that norm, not element by element (small components carry
    # large relative errors). The bound is per panel, not on the total, and
    # max_iter can accept a panel that misses it, so no tolerance follows from
    # it: 5e-2 is an empirical margin. Measured 2026-11-17 on CPU, seeds 0-5 for
    # the meshes (this test uses 0): relative L2 difference at most 7.1e-3
    # (extend) and 2.5e-6 (cut). Both runs are seeded, so the result is
    # deterministic on a given machine.
    diff = (out.grad_integral - expected.grad_integral).norm()
    assert diff <= 5e-2 * expected.grad_integral.norm()


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
                raise torch.OutOfMemoryError(
                    f"simulated out-of-memory error at {t.shape[0]} points"
                )
        return evaluate_integrand_sum(terms, t, path, **kwargs)

    monkeypatch.setattr(
        "popcornn.tools.integrator.evaluate_integrand_sum", evaluate_with_limit
    )
    return batch_sizes


@pytest.mark.parametrize(
    ("integrand", "integrate_gradient"),
    [("main", True), ("loss", True), ("main", False)],
)
def test_oom_batch_reduction_persists(
    muller_brown_setup, monkeypatch, integrand, integrate_gradient
):
    """An out-of-memory error makes padaquad shrink the batch and retry. The
    reduced size comes back in the state, which each solver passes to its next
    call, so the error must not recur on the next ``integrate_path``.
    ``'main'`` exercises ``_solver`` (in both gradient modes; with
    ``integrate_gradient=False`` padaquad takes the gradient itself),
    ``'loss'`` exercises ``_loss_solver`` on its own.
    """
    path, device, dtype = muller_brown_setup
    limit = 50  # more than gk21's 21 nodes per panel, so one panel always fits
    batch_sizes = _simulate_oom(monkeypatch, limit, integrand)
    integrator = PathIntegrator(
        method="gk21",
        path_integrand_names="pvre",
        rtol=1e-2,
        atol=1e-2,
        max_batch=400,
        integrate_gradient=integrate_gradient,
        track_loss=True,
        device=device,
        dtype=dtype,
    )

    integrator.integrate_path(path)
    assert max(batch_sizes) > limit  # the first call hit the simulated error

    batch_sizes.clear()
    integrator.integrate_path(path)
    assert max(batch_sizes) <= limit


@pytest.mark.parametrize(
    "memory_options",
    [
        {"trade_speed_for_memory": False},
        {"evaluation_buffer": None},
        {"evaluation_buffer": 10_000},
        {"result_device": None},
        {"record_evaluations": False},
        {"record_evaluations": False, "track_ts": False},
        {"runtime_checks": True},
    ],
    ids=[
        "no_trade",
        "no_buffer",
        "small_buffer",
        "result_on_device",
        "no_record_ts",
        "no_record",
        "runtime_checks",
    ],
)
def test_memory_options_leave_results_unchanged(muller_brown_setup, memory_options):
    """padaquad's memory options change only where and when integrand values
    are held, never the evaluated points, the mesh or the integral, which
    padaquad states holds bit for bit. Each option is compared with popcornn's
    defaults (``trade_speed_for_memory=True``, a 5 GiB ``evaluation_buffer``
    that these passes never fill) over two warm-started calls.
    ``max_batch=46`` spreads a pass over several calls of the integrand, so
    ``no_buffer`` and ``small_buffer`` (10 kB, less than one call's values)
    compute the quadrature after every call. With
    ``track_ts`` on, the values are recorded even with
    ``record_evaluations=False``, so the TS samples must match too
    (``no_record_ts``); ``no_record`` turns ``track_ts`` off so that nothing
    is recorded; its next mesh is built differently, so only its first call
    is compared. ``runtime_checks=True`` (popcornn's default is False) only
    adds checks, so it must not change the results either."""
    path, device, dtype = muller_brown_setup
    params = {
        "method": "gk21",
        "path_integrand_names": "pvre",
        "rtol": 1e-2,
        "atol": 1e-2,
        "max_batch": 46,
        "track_loss": True,
        "track_ts": True,
        "device": device,
        "dtype": dtype,
    }
    runs = []
    for options in ({}, memory_options):
        torch.manual_seed(0)  # the first mesh is random
        integrator = PathIntegrator(**(params | options))
        runs.append([integrator.integrate_path(path) for _ in range(2)])
    if "track_ts" in memory_options:
        # Nothing recorded: padaquad builds the next call's mesh with
        # merge_rule="power_law" instead of "predicted", so only the first
        # call is the same.
        assert runs[1][0].samples is None
        runs = [run[:1] for run in runs]
    for default, changed in zip(*runs, strict=True):
        assert torch.equal(changed.grad_integral, default.grad_integral)
        assert torch.equal(changed.loss, default.loss)
        if changed.samples is None:
            continue
        for field in ("time", "energies", "dEdt"):
            assert torch.equal(
                getattr(changed.samples, field), getattr(default.samples, field)
            )
        assert changed.samples.time.numel() > 0


@pytest.mark.parametrize("runtime_checks", [None, True])
def test_device_and_runtime_checks_reach_padaquad(monkeypatch, runtime_checks):
    """The GPU index of ``device`` must reach padaquad, so a ``cuda:1`` run
    integrates on GPU 1, and ``runtime_checks`` must reach both solvers,
    False when not given (padaquad's own default is True). Runs without a
    GPU: only the solver construction is intercepted.
    """
    calls = []

    def record_options(*args, **kwargs):
        calls.append((kwargs["device"], kwargs["runtime_checks"]))
        return object()

    monkeypatch.setattr("popcornn.tools.integrator.adaptive_quadrature", record_options)
    given = {} if runtime_checks is None else {"runtime_checks": runtime_checks}
    PathIntegrator(
        path_integrand_names="pvre",
        max_batch=500,  # never integrates; pinned only for the conftest guard
        device=torch.device("cuda", 1),
        dtype=torch.float64,
        **given,
    )
    assert calls == [(torch.device("cuda", 1), bool(runtime_checks))] * 2
