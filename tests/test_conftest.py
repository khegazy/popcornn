"""Tests of the shared test machinery in ``tests/conftest.py``."""

from __future__ import annotations

import re

import pytest
import torch
from padaquad import adaptive_quadrature

from popcornn.tools import PathIntegrator
from tests import IMAGES_DIR, TEST_SEED, starting_mesh


@pytest.mark.parametrize("repeat", [1, 2])
def test_each_test_starts_from_the_same_random_state(repeat):
    """Every test starts from torch.manual_seed(TEST_SEED), whatever ran before.
    The second case fails without the conftest fixture, because the first case
    has already advanced the global generator."""
    expected = torch.rand(4, generator=torch.Generator().manual_seed(TEST_SEED))
    assert torch.equal(torch.rand(4), expected)


def test_unpinned_max_batch_is_refused(cheap_integrator_params):
    """Every test pins max_batch: without it padaquad searches for a batch size
    by growing it until it runs out of memory, which depends on the machine."""
    with pytest.raises(RuntimeError, match="unpinned_max_batch"):
        PathIntegrator(
            **{
                **cheap_integrator_params,
                "max_batch": None,
                "path_integrand_names": "pvre",
            }
        )


@pytest.mark.unpinned_max_batch
def test_unpinned_max_batch_allowed_when_marked(cheap_integrator_params):
    """A test that tests padaquad's batch-size search opts out with the marker."""
    PathIntegrator(
        **{**cheap_integrator_params, "max_batch": None, "path_integrand_names": "pvre"}
    )


def test_device_and_dtype_fixtures(device, dtype):
    """The shared fixtures give every test the same device and dtype cases."""
    assert device.type in ("cpu", "cuda")
    assert dtype in (torch.float32, torch.float64)


def test_cheap_integrator_params_build_a_guarded_integrator(cheap_integrator_params):
    """The cheap settings pin max_batch (so the guard accepts them) and turn on
    padaquad's runtime checks, which popcornn leaves off by default."""
    assert cheap_integrator_params["max_batch"] is not None
    assert cheap_integrator_params["runtime_checks"] is True
    PathIntegrator(**{**cheap_integrator_params, "path_integrand_names": "pvre"})


@pytest.mark.parametrize("seed", [0, 3])
def test_starting_mesh_is_padaquads_mesh_for_the_same_seed(seed):
    """starting_mesh(seed) is the starting mesh padaquad draws after
    torch.manual_seed(seed), so a fixed-mesh run samples the same meshes as a
    seeded run without one. Checked through padaquad's public API: both
    integrations must start from as many subintervals and evaluate the same
    nodes."""

    def f(t):
        return torch.sin(3 * t)  # any smooth integrand

    lo = torch.zeros(1, dtype=torch.float64)
    hi = torch.ones(1, dtype=torch.float64)
    solver = adaptive_quadrature(
        method="gk7", rtol=1e-6, atol=1e-10, device="cpu", max_batch=500
    )
    given, _ = solver.integrate(
        f, mesh=starting_mesh(seed), mesh_init=lo, mesh_final=hi, take_gradient=False
    )
    torch.manual_seed(seed)
    drawn, _ = solver.integrate(f, mesh_init=lo, mesh_final=hi, take_gradient=False)
    assert given.n_start_steps == drawn.n_start_steps
    assert torch.equal(given.nodes, drawn.nodes)


def test_no_test_reads_data_relative_to_the_working_directory():
    """Test data is opened through IMAGES_DIR, so pytest works from any
    directory. A literal path into the images folder, relative to the
    working directory, works only when pytest starts in one particular place."""
    relative = re.compile(r"""["'](tests/)?images/""")
    offenders = [
        f"{module.name}:{number}"
        for module in sorted(IMAGES_DIR.parent.glob("*.py"))
        for number, line in enumerate(module.read_text().splitlines(), start=1)
        if relative.search(line)
    ]
    assert offenders == [], f"working-directory-relative data paths: {offenders}"
