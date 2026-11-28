"""Are ``PathIntegrator``'s default tolerances accurate enough on a real molecule?

Compares the default tolerances (``method="gk7"``, ``rtol=1.0``, ``atol=0.0``,
``max_iter=5``) with a tight reference on the T1x reaction (C3H8O2, 13 atoms;
the same reaction as ``examples/configs/gg3.xyz``) with the Morse potential and
the ``pvre`` integrand. Three quantities are compared:

- the direction of the integrated gradient (cosine with the reference),
- its size (relative error ``|g - g_ref| / |g_ref|``),
- the path after a short optimization, relative to how far the optimization
  moved it.

Morse stands in for UMA, which the examples use but whose weights are gated.
On T1x, Morse has its energy maximum at an endpoint, so these tests judge the
gradient and the path, not the transition state.

Thresholds and the reasoning behind them are in
tests_ongoing/test_plans/00_plan_shared_test_infrastructure.md (Fact 9, Task
5b); the values in the comments below were measured 2026-11-27 with padaquad
2.0.0 on CPU in float64, from the fixed starting meshes ``starting_mesh(0..9)``.
"""

from __future__ import annotations

import pytest
import torch

from popcornn import Popcornn
from popcornn.tools import PathIntegrator
from tests import IMAGES_DIR, TEST_SEED, make_path, starting_mesh

pytestmark = [
    pytest.mark.slow,  # the tight reference takes 10-45 s on a loaded login node
    pytest.mark.skip(
        reason="Skipped on request until the default tolerances are decided (plan 00, Task 5b)."
    ),
]

IMAGES = str(IMAGES_DIR / "T1x.xyz")
DEVICE = torch.device("cpu")
DTYPE = torch.float64
PATH_PARAMS = {"name": "mlp", "width": 8, "depth": 2}
PATH_SEED = 0  # MLP initialization
# Seeds of starting_mesh: each is a different fixed starting mesh for the same
# path, so the comparison covers several meshes without padaquad's random draws.
MESH_SEEDS = range(10)

# Settings shared by every integration:
# - integrate_gradient=False: at width 128 it gave the same gradient errors as
#   the default True (Fact 9) and is about 8x cheaper (#89).
# - max_batch=4000: every node of every integration here fits in one batch, and
#   memory grows linearly in the batch with integrate_gradient=False.
COMMON = {
    "path_integrand_names": "pvre",
    "integrate_gradient": False,
    "max_batch": 4000,
    "runtime_checks": True,  # tests raise on NaN/Inf (popcornn's default is False)
}
# Tight reference. Its integrand has corners (where dE/dt changes sign), so even
# these tolerances leave a relative gradient difference of up to 1.6e-3 between
# starting meshes 0-5.
TIGHT = {"method": "gk21", "rtol": 1e-8, "atol": 1e-12, "max_iter": 50}
# Tight settings for the optimization comparison (fewer iterations, so looser).
TIGHT_OPTIMIZATION = {"method": "gk21", "rtol": 1e-6, "atol": 1e-8, "max_iter": 20}
OPTIMIZER_PARAMS = {"optimizer": {"name": "adam", "lr": 1e-2}}
NUM_ITERATIONS = 20
T_RECORD = torch.linspace(0.0, 1.0, 21, dtype=DTYPE)

# Thresholds (accepted 2026-11-17), with the values measured at the defaults.
REFERENCE_AGREEMENT_MAX = 2e-3  # measured 7.8e-4 (meshes 0 vs 3); <= 1.6e-3 over 0-5
COSINE_MIN = 0.99  # measured minimum over meshes 0-9: 0.9997
SIZE_ERROR_MAX = 0.2  # measured median 0.13, maximum 1.26
PATH_RATIO_MAX = 0.05  # measured 0.152 (mesh TEST_SEED)


def _integrated_gradient(mesh_seed: int, **integrator_params) -> torch.Tensor:
    """Integrated pvre gradient over all path parameters, for one starting mesh.

    Parameters
    ----------
    mesh_seed : int
        Seed of ``starting_mesh``, this integration's fixed starting mesh.
    **integrator_params
        Extra ``PathIntegrator`` arguments; none means the default tolerances.

    Returns
    -------
    torch.Tensor
        The flattened gradient, shape [number of trainable parameters].
    """
    torch.manual_seed(PATH_SEED)
    path = make_path(IMAGES, "morse", DEVICE, DTYPE, **PATH_PARAMS)
    integrator = PathIntegrator(
        **COMMON,
        **integrator_params,
        initial_mesh=starting_mesh(mesh_seed),
        device=DEVICE,
        dtype=DTYPE,
    )
    integrator.integrate_path(path)
    return torch.cat(
        [p.grad.flatten() for p in path.parameters() if p.grad is not None]
    )


def _optimize(**integrator_params) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the short optimization and return the path before and after it.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Positions at ``T_RECORD`` before and after optimizing, each [21, 39].
    """
    mep = Popcornn(
        images=IMAGES,
        path_params=PATH_PARAMS,
        device="cpu",
        dtype="float64",
        seed=PATH_SEED,
        quiet=True,
    )
    start = mep.path(T_RECORD).positions.detach().clone()
    mep.optimize_path(
        {
            "potential_params": {"name": "morse"},
            "integrator_params": {
                **COMMON,
                "initial_mesh": starting_mesh(TEST_SEED),
                **integrator_params,
            },
            "optimizer_params": OPTIMIZER_PARAMS,
            "num_optimizer_iterations": NUM_ITERATIONS,
        }
    )
    return start, mep.path(T_RECORD).positions.detach()


@pytest.fixture(scope="module")
def reference_gradient():
    return _integrated_gradient(0, **TIGHT)


@pytest.fixture(scope="module")
def default_gradients():
    return [_integrated_gradient(seed) for seed in MESH_SEEDS]


def test_tight_reference_is_converged(reference_gradient):
    """The reference must not depend on its starting mesh, or the comparisons below mean nothing."""
    other = _integrated_gradient(3, **TIGHT)
    difference = float((other - reference_gradient).norm() / reference_gradient.norm())
    assert difference <= REFERENCE_AGREEMENT_MAX


def test_default_gradient_direction(reference_gradient, default_gradients):
    """At the default tolerances the gradient points the right way."""
    cosines = [
        float(torch.nn.functional.cosine_similarity(g, reference_gradient, dim=0))
        for g in default_gradients
    ]
    assert min(cosines) >= COSINE_MIN, f"cosines per starting mesh: {cosines}"


@pytest.mark.xfail(
    strict=True,
    reason="Default tolerances get the gradient size wrong by up to 126% (plan 00, Fact 9).",
)
def test_default_gradient_size(reference_gradient, default_gradients):
    """At the default tolerances the gradient has the right size and direction."""
    errors = [
        float((g - reference_gradient).norm() / reference_gradient.norm())
        for g in default_gradients
    ]
    assert max(errors) <= SIZE_ERROR_MAX, f"relative errors per starting mesh: {errors}"


@pytest.mark.xfail(
    strict=True,
    reason="After 20 steps the default-tolerance path is 15% off the tight one (plan 00, Fact 9).",
)
def test_default_optimization_matches_tight():
    """A short optimization at the default tolerances ends near the tight-tolerance result."""
    start, tight = _optimize(**TIGHT_OPTIMIZATION)
    _, default = _optimize()
    ratio = float((default - tight).norm() / (tight - start).norm())
    assert ratio <= PATH_RATIO_MAX, (
        f"|x_default - x_tight| / |x_tight - x_start| = {ratio:.3g}"
    )
