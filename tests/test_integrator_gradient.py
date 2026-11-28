"""The ``pvre`` path integral and its gradient against exact values.

What is compared
    ``PathIntegrator`` integrates the ``pvre`` loss L = ∫ |dE/dt| dt of an MLP path
    and its gradient with respect to every path parameter. Both are compared with
    exact values that use no quadrature: L is the total variation of E along the
    path, and its gradient is a sum over the turning points t_c of E(t), where
    dE/dt changes sign (plan 01, eqs. 1 and 3; the spec's eqs. 2 and 4 at zero
    weights):

        L = Σ_k |E(t_{k+1}) - E(t_k)|,
        ∂L/∂θ = Σ_c (s⁻_c - s⁺_c) ∇E(x(t_c)) · ∂x/∂θ(t_c),

    with s⁻_c and s⁺_c the signs of dE/dt just before and after t_c. The turning
    points are found with a root-finder on a dense grid.

Why two weight sets
    - Zero weights: the path is the straight line and only the last-layer bias b
      has a gradient (the spec's eq. 4). The sphere's closed form and the
      pre-0.2.0 Müller-Brown numbers fix ∂L/∂b independently of popcornn's path.
    - Random weights (PyTorch's default initialization, seeded with TEST_SEED,
      rounded to float32): every parameter has a gradient, so the check covers
      the hidden layers, the scatter of the gradient into ``.grad`` and
      ``grad_norm``. Central finite differences of L confirm eq. 3 there.

Where the limits come from
    Each case has its own limits, 10x its worst error over starting_mesh(0..9) on
    CPU and CUDA (plan 01, "Tolerance design" and Fact 2). Two padaquad failure
    modes set the largest ones: a turning point in a panel's unsampled edge gap
    (#104; Fact 4) and, in float32, a rounding floor that accepts a panel holding
    a turning point (Fact 5). Plan:
    tests_ongoing/test_plans/01_integral_and_gradient_implementation_plan.md.
    The 10x margin holds only on these 10 meshes. On other starting meshes a
    turning point in an edge gap gave errors up to 1.7e-4, in every setting
    including rtol=1e-12, with padaquad reporting convergence (sphere at random
    weights, starting_mesh(23); plan 01, Fact 4). Do not add meshes without
    re-measuring.

What no case here can see
    - A sign error in the velocity, since |F·(-v)| = |F·v|.
    - A bug in ``MLPpath`` that keeps the endpoints pinned (for example a wrong
      hidden layer) in the integration cases, because the reference uses the same
      path; the zero-weight reference checks catch bugs in ∂x/∂b. A bug that
      unpins an endpoint is caught: eq. 3 assumes ∂x/∂θ = 0 there.
    - Confusing the main and loss solvers of ``PathIntegrator``.
    - Other random weights: the random-weight cases are one draw (seed TEST_SEED,
      PyTorch's default initialization, width 128). A more strongly curved path,
      with more turning points, is not tested.

How to re-measure
    If a case fails after a padaquad update, run the slow mesh sweep
    (``pytest tests/test_integrator_gradient.py -k mesh``) and check whether
    quadrature nodes moved next to the turning points (#104) before suspecting
    popcornn.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pytest
import torch
from scipy.optimize import brentq
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from popcornn.tools import PathIntegrator
from tests import IMAGES_DIR, REFERENCE_TOL, TEST_SEED, make_path, starting_mesh

if TYPE_CHECKING:
    from popcornn.paths import MLPpath

# Test system: (image file, potential name), as make_path takes them.
SYSTEMS = {
    "sphere": (str(IMAGES_DIR / "sphere.json"), "sphere"),
    "muller_brown": (str(IMAGES_DIR / "muller_brown.json"), "muller_brown"),
    "wolfe": (str(IMAGES_DIR / "wolfe.json"), "wolfe_schlegel"),
    "LJ13": (str(IMAGES_DIR / "LJ13.xyz"), "lennard_jones"),
}
WEIGHTS = ("zero", "random")

# (L - grid total variation) / L. Measured at most 3.9e-9 (muller_brown, zero
# weights); the lower bound allows rounding (LJ13 +2.3e-15). Plan 01, Fact 1.
GRID_TV_LIMITS = (-1e-12, 1e-8)
# Pre-0.2.0 values from tests_ongoing/test_integrators.py, zero weights. They carry
# the old integrator's error, including the edge-gap blind spot (#104): measured
# 3.8e-8, 1.9e-7 and below 1e-10 for L, and 1.2e-6 for the muller_brown gradient.
LEGACY_LOSS_RTOL = 1e-6
LEGACY_GRAD_RTOL = 1e-5
LEGACY = {
    "muller_brown": (
        234.27849339874268,
        torch.tensor([36.29373537, 18.90991724], dtype=torch.float64),
    ),
    "wolfe": (131.19240801706766, None),  # its old gradient is off by about 5e-5
    "LJ13": (18.8414107735614, None),
}
# Central differences of eq. 1 against eq. 3, relative to |∂L/∂θ|: worst 9.3e-11
# (wolfe) at h = 1e-4, which minimizes truncation plus rounding. Plan 01, Fact 3.
FD_STEP = 1e-4
FD_GRAD_TOL = 1e-9
FD_DIRECTIONS = 3
GRID_POINTS = 20001  # brackets the turning points; LJ13's lie within 1e-9 of the ends


class ExactReference(NamedTuple):
    """Exact ``pvre`` loss and gradient of one path (plan 01, eqs. 1 and 3)."""

    loss: float
    grad: torch.Tensor  # [D], float64, CPU, in path.parameters() order
    turning_times: tuple[float, ...]
    grid_total_variation: float  # Σ |ΔE| over GRID_POINTS evenly spaced times
    state_dict: dict[str, torch.Tensor]  # the path's weights, float64, CPU


def _turning_point_terms(path: MLPpath) -> ExactReference:
    """L from eq. (1) and ∂L/∂θ from eq. (3) for a float64 CPU path, without quadrature.

    Turning points are bracketed by sign changes of dE/dt on a grid and refined
    with ``brentq``. Every turning point is kept, including those next to the
    endpoints, which ``BasePath.ts_search`` would drop.

    Preconditions, not all checkable here: the path is float64 on the CPU
    (checked); its endpoints are pinned, so ∂x/∂θ = 0 at t = 0 and 1 (eq. 3
    drops endpoint terms); and its turning points are simple sign changes more
    than one grid step (1 / (GRID_POINTS - 1)) apart. Two turning points inside
    one grid cell are missed, and the grid total variation cannot reveal it,
    since it uses the same grid.
    """
    params = list(path.parameters())
    if any(p.dtype != torch.float64 or p.device.type != "cpu" for p in params):
        msg = "_turning_point_terms needs a float64 path on the CPU."
        raise ValueError(msg)

    def evaluate(t):
        """E, dE/dt and F at times t (a float or 1-D array), all detached."""
        t = torch.as_tensor(np.atleast_1d(t), dtype=torch.float64).requires_grad_(True)
        out = path(t, return_velocities=True, return_energies=True, return_forces=True)
        # The same dE/dt as PathIntegrator.
        edot = -(out.forces * out.velocities).sum(-1)
        return (
            out.energies[:, 0].detach().numpy(),
            edot.detach().numpy(),
            out.forces.detach(),
        )

    grid = np.linspace(0.0, 1.0, GRID_POINTS)
    energies, edot, _ = evaluate(grid)
    # Skip exact zeros (the sphere at zero weights has dE/dt = 0 at t = 1/2): a turning
    # point lies between consecutive nonzero samples of opposite sign.
    kept = np.flatnonzero(edot)
    signs = np.sign(edot[kept])
    turning_times = []
    grad = torch.zeros(parameters_to_vector(params).numel(), dtype=torch.float64)
    for k in np.flatnonzero(signs[:-1] != signs[1:]):
        t_c = brentq(
            lambda s: float(evaluate(s)[1][0]),
            grid[kept[k]],
            grid[kept[k + 1]],
            xtol=1e-15,
            rtol=4 * np.finfo(float).eps,  # scipy's minimum
        )
        turning_times.append(t_c)
        # Plan 01, eq. (3); the spec's eq. (4) at zero weights:
        # (s⁻ - s⁺) ∇E(x(t_c)) · ∂x/∂θ(t_c), with ∇E = -F held fixed.
        positions = path.get_positions(torch.tensor([[t_c]], dtype=torch.float64))
        grad_energy = -evaluate(t_c)[2]
        terms = torch.autograd.grad((grad_energy * positions).sum(), params)
        grad += float(signs[k] - signs[k + 1]) * parameters_to_vector(terms)
    if not turning_times:
        msg = f"No turning point of E along the {type(path.potential).__name__} path."
        raise ValueError(msg)
    knots = np.array([0.0, *turning_times, 1.0])
    return ExactReference(
        loss=float(np.abs(np.diff(evaluate(knots)[0])).sum()),  # eq. (1)
        grad=grad,
        turning_times=tuple(turning_times),
        grid_total_variation=float(np.abs(np.diff(energies)).sum()),
        state_dict={
            name: value.detach().clone() for name, value in path.state_dict().items()
        },
    )


@functools.cache
def _exact_reference(system: str, weights: str) -> ExactReference:
    """The exact reference for ``system``'s MLP path with zero or random weights.

    Random weights are PyTorch's default initialization after
    ``torch.manual_seed(TEST_SEED)``, rounded to float32 so float32 and float64
    runs use exactly the same weights.
    """
    if weights not in WEIGHTS:
        msg = f"weights must be one of {WEIGHTS}, got {weights!r}."
        raise ValueError(msg)
    # Cached, so first computed inside whichever test runs first: seed a forked
    # CPU generator so the weights do not depend on test order, and leave every
    # global generator as the conftest set it (plan 01, deviation 11). MLPpath
    # creates its layers on the CPU, so only the CPU generator is seeded;
    # torch.manual_seed would also reseed the CUDA generators, which fork_rng
    # (devices=[]) does not restore.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(TEST_SEED)
        path = make_path(*SYSTEMS[system], torch.device("cpu"), torch.float64)
    with torch.no_grad():
        for p in path.parameters():
            p.copy_(torch.zeros_like(p) if weights == "zero" else p.float())
    return _turning_point_terms(path)


@pytest.mark.parametrize(
    ("system", "weights"),
    [
        pytest.param(
            system,
            weights,
            id=f"{system}-{weights}",
            # LJ13's finite differences take about 12 s of CPU time.
            marks=pytest.mark.slow if (system, weights) == ("LJ13", "random") else (),
        )
        for system in SYSTEMS
        for weights in WEIGHTS
    ],
)
def test_exact_reference(system: str, weights: str) -> None:
    """The exact reference is right: total variation on a dense grid, the sphere's
    closed form and the pre-0.2.0 numbers at zero weights, and central finite
    differences of L at random weights."""
    ref = _exact_reference(system, weights)
    low, high = GRID_TV_LIMITS
    assert low * ref.loss <= ref.loss - ref.grid_total_variation <= high * ref.loss
    if weights == "zero":
        bias_grad = ref.grad[-ref.state_dict["mlp.2.bias"].numel() :]
        # Spec eq. (4): only b has a gradient.
        assert not ref.grad[: -bias_grad.numel()].any()
        # Sphere: E = |x|^2 from (-1, 1) to (1, 1), one minimum at t = 1/2.
        if system == "sphere":
            # Relative to the largest reference value, 2 and 1 here.
            tol = REFERENCE_TOL[torch.float64]
            assert ref.turning_times == pytest.approx((0.5,), abs=tol)
            assert ref.loss == pytest.approx(2.0, abs=2.0 * tol)
            torch.testing.assert_close(
                bias_grad,
                torch.tensor([0.0, -1.0], dtype=torch.float64),
                rtol=0,
                atol=tol,
            )
        if system in LEGACY:
            legacy_loss, legacy_grad = LEGACY[system]
            assert ref.loss == pytest.approx(legacy_loss, rel=LEGACY_LOSS_RTOL)
            if legacy_grad is not None:
                assert (
                    float((bias_grad - legacy_grad).norm() / bias_grad.norm())
                    <= LEGACY_GRAD_RTOL
                )
    else:
        # Eq. (3) is the derivative of eq. (1): central differences along random directions.
        path = make_path(*SYSTEMS[system], torch.device("cpu"), torch.float64)
        path.load_state_dict(ref.state_dict)
        theta = parameters_to_vector(path.parameters()).detach()
        generator = torch.Generator().manual_seed(TEST_SEED)
        for _ in range(FD_DIRECTIONS):
            d = torch.randn(theta.numel(), generator=generator, dtype=torch.float64)
            d /= d.norm()
            vector_to_parameters(theta + FD_STEP * d, path.parameters())
            loss_plus = _turning_point_terms(path).loss
            vector_to_parameters(theta - FD_STEP * d, path.parameters())
            loss_minus = _turning_point_terms(path).loss
            fd_error = abs(
                (loss_plus - loss_minus) / (2 * FD_STEP) - float(ref.grad @ d)
            )
            assert fd_error <= FD_GRAD_TOL * float(ref.grad.norm()), f"{fd_error:.2e}"


# Largest single call is 231 points; 500**2 * 128 = 3.2e7 (plan 00 memory rule).
MAX_BATCH = 500
MESH_SEEDS = range(10)  # CI runs TEST_SEED; the other meshes are slow cases
# The float64 loss of muller_brown and wolfe has one error source, a turning point next to
# an endpoint (plan 01, Fact 4; #104). Its worst case is the same in both float64
# settings (zero weights 3.8e-8 and 1.9e-7, random weights 3.3e-8 and 1.8e-7), so the
# limit is the same in both settings, at both weights.
FLOAT64_ENDPOINT_GAP_LOSS_LIMIT = {"muller_brown": 5e-7, "wolfe": 2e-6}
# Limits on the relative gradient and loss errors, [weights][system]: smallest
# {1,2,5}x10^k >= 10x that case's worst error over starting_mesh(0..9), CPU and CUDA,
# padaquad 2.0.0, 2026-11-27 (plan 01, Fact 2).
SETTINGS = {
    # wolfe: a turning point in an interior panel's edge gap (#104); muller_brown: one
    # next to t = 0 (#104).
    "grad_in_integrand-float64": {
        "dtype": torch.float64,
        "integrator": {
            "integrate_gradient": True,
            "method": "gk21",
            "rtol": 1e-8,
            "atol": 1e-10,
            "max_iter": 50,
            "loss_rtol": 0.0,
            "loss_atol": 1e-9,
        },
        "max_grad_error": {
            "zero": {"sphere": 1e-7, "muller_brown": 5e-6, "wolfe": 1e-4, "LJ13": 5e-8},
            "random": {
                "sphere": 5e-8,
                "muller_brown": 5e-6,
                "wolfe": 2e-4,
                "LJ13": 1e-6,
            },
        },
        "max_loss_error": {
            "zero": {"sphere": 5e-9, "LJ13": 2e-9, **FLOAT64_ENDPOINT_GAP_LOSS_LIMIT},
            "random": {
                "sphere": 5e-9,
                "LJ13": 1e-10,
                **FLOAT64_ENDPOINT_GAP_LOSS_LIMIT,
            },
        },
    },
    # As above, plus float32 rounding; the wolfe loss at random weights is set by
    # padaquad's float32 rounding floor accepting a panel that holds the maximum
    # (plan 01, Fact 5).
    "grad_in_integrand-float32": {
        "dtype": torch.float32,
        "integrator": {
            "integrate_gradient": True,
            "method": "gk21",
            "rtol": 1e-6,
            "atol": 1e-8,
            "max_iter": 20,
            "loss_rtol": 0.0,
            "loss_atol": 1e-6,
        },
        "max_grad_error": {
            "zero": {"sphere": 1e-5, "muller_brown": 2e-5, "wolfe": 1e-4, "LJ13": 1e-5},
            "random": {
                "sphere": 1e-5,
                "muller_brown": 1e-5,
                "wolfe": 2e-4,
                "LJ13": 1e-5,
            },
        },
        "max_loss_error": {
            "zero": {"sphere": 2e-5, "muller_brown": 2e-6, "wolfe": 5e-6, "LJ13": 5e-6},
            "random": {
                "sphere": 5e-6,
                "muller_brown": 5e-6,
                "wolfe": 5e-5,
                "LJ13": 1e-5,
            },
        },
    },
    # padaquad controls the error of the loss here, not of the gradient, whose integrand
    # jumps at every turning point (plan 01, deviation 5). No separate loss solver.
    "grad_of_integral-float64": {
        "dtype": torch.float64,
        "integrator": {
            "integrate_gradient": False,
            "method": "gk21",
            "rtol": 1e-12,
            "atol": 1e-14,
            "max_iter": 60,
        },
        "max_grad_error": {
            "zero": {"sphere": 5e-6, "muller_brown": 5e-5, "wolfe": 5e-4, "LJ13": 5e-6},
            "random": {
                "sphere": 1e-5,
                "muller_brown": 1e-4,
                "wolfe": 5e-4,
                "LJ13": 5e-6,
            },
        },
        "max_loss_error": {
            "zero": {"sphere": 2e-11, "LJ13": 5e-11, **FLOAT64_ENDPOINT_GAP_LOSS_LIMIT},
            "random": {
                "sphere": 5e-12,
                "LJ13": 5e-11,
                **FLOAT64_ENDPOINT_GAP_LOSS_LIMIT,
            },
        },
    },
}


def _lj13_gradient_in_integrand(system: str, setting: str) -> bool:
    """LJ13 with the gradient in the integrand: 1-3 min of CPU time per case (#89)."""
    return system == "LJ13" and SETTINGS[setting]["integrator"]["integrate_gradient"]


def _is_slow(system: str, setting: str, mesh_seed: int) -> bool:
    """Meshes other than TEST_SEED form the local-only sweep, and the expensive
    LJ13 cases are slow on every mesh."""
    return mesh_seed != TEST_SEED or _lj13_gradient_in_integrand(system, setting)


@pytest.mark.parametrize(
    ("system", "weights", "setting", "mesh_seed"),
    [
        pytest.param(
            system,
            weights,
            setting,
            mesh_seed,
            id=f"{system}-{weights}-{setting}-mesh{mesh_seed}",
            marks=pytest.mark.slow if _is_slow(system, setting, mesh_seed) else (),
        )
        for system in SYSTEMS
        for weights in WEIGHTS
        for setting in SETTINGS
        for mesh_seed in MESH_SEEDS
    ],
)
def test_pvre_loss_and_gradient_match_exact(
    system: str, weights: str, setting: str, mesh_seed: int, device: torch.device
) -> None:
    """``PathIntegrator``'s pvre loss and gradient match the exact reference.

    One case integrates ``system``'s MLP path at zero or random weights with one
    gradient mode and dtype (``setting``) from ``starting_mesh(mesh_seed)``. It checks
    the relative gradient error over all parameters and the relative loss error
    against that case's own limits (plan 01, "Tolerance design"; #104), the device and
    dtype of the outputs, exact zeros off the last-layer bias at zero weights, and that
    ``grad_norm`` is the L2 norm of the whole gradient.
    """
    s = SETTINGS[setting]
    dtype = s["dtype"]
    if (
        device.type == "cpu"
        and _lj13_gradient_in_integrand(system, setting)
        and (weights == "random" or mesh_seed != TEST_SEED)
    ):
        pytest.skip("1-3 min of CPU time per case (#89); the CUDA cases sweep these")
    ref = _exact_reference(system, weights)
    path = make_path(*SYSTEMS[system], device, dtype)
    path.load_state_dict(ref.state_dict)  # copies cast to the path's dtype and device
    out = PathIntegrator(
        path_integrand_names="pvre",
        track_loss=True,
        max_batch=MAX_BATCH,
        runtime_checks=True,
        initial_mesh=starting_mesh(mesh_seed),
        device=device,
        dtype=dtype,
        **s["integrator"],
    ).integrate_path(path)

    for name, p in path.named_parameters():
        assert p.grad is not None, name
    grad = torch.cat([p.grad.flatten() for p in path.parameters()])
    assert grad.dtype == dtype
    assert grad.device.type == device.type
    assert out.loss.dtype == dtype
    assert out.loss.device.type == device.type
    grad_error = float((grad.double().cpu() - ref.grad).norm() / ref.grad.norm())
    assert grad_error <= s["max_grad_error"][weights][system], (
        f"relative gradient error {grad_error:.2e}"
    )
    loss_error = abs(float(out.loss) - ref.loss) / ref.loss
    assert loss_error <= s["max_loss_error"][weights][system], (
        f"relative loss error {loss_error:.2e}"
    )
    # Spec eq. (4): exactly zero through a zero weight or GELU(0) = 0.
    if weights == "zero":
        bias = path.mlp[-1].bias
        for name, p in path.named_parameters():
            if p is not bias:
                assert not p.grad.any(), name
    # grad_norm, which PathOptimizer's stop test reads, is the L2 norm of the whole flat gradient.
    torch.testing.assert_close(out.grad_norm, grad.norm())
