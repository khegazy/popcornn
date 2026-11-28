"""Two-stage atomic end-to-end test of the ``Popcornn`` pipeline.

Runs a repel + geodesic stage and then a Morse + pvre + geodesic stage on the
17-atom molecule in ``hard.xyz``. It checks how the parts connect: ``.xyz``
input, two stages training one path, the ASE path output, and the TS output.
Part A runs the stages as two ``optimize_path`` calls so the repel stage can be
measured on its own; part B runs them in the single call users make and checks
that it gives the same path. The test deliberately does not compare against a
stored trajectory or check numerical accuracy (the integral and potential
tests do that), which is why it uses the cheap integrator settings from
``conftest``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read, write

from popcornn import Popcornn
from popcornn.potentials.morse import MorsePotential
from tests import IMAGES_DIR

XYZ = str(IMAGES_DIR / "hard.xyz")  # process_images needs a str
# Built twice with these arguments (parts A and B). Popcornn's default seed=0
# reseeds torch, so both builds start from the same MLP weights.
POPCORNN_KWARGS = {
    "images": XYZ,
    "path_params": {"name": "mlp", "width": 8, "depth": 2},
    "num_record_points": 11,
    "device": "cpu",
    "dtype": "float64",
}
# As in both stages of examples/configs/gg3.yaml. No threshold, so every stage
# stops at its iteration cap and its flag is always False.
OPTIMIZER_PARAMS = {"optimizer": {"name": "adam", "lr": 0.01}}
N_ITERATIONS = 2


def _min_interatomic_distance(frames: list[Atoms]) -> float:
    """Return the smallest distance between any two atoms over all frames, in Å."""
    return min(
        float(np.min(f.get_all_distances()[np.triu_indices(len(f), k=1)]))
        for f in frames
    )


def test_two_stage_atomic_run_returns_ase_path_and_ts(cheap_integrator_params):
    # Cheap settings (conftest) plus the fast gradient mode (#89).
    integrator_params = {**cheap_integrator_params, "integrate_gradient": False}
    # Stage 1: cheap clash removal. Stage 2: Morse stands in for an MLIP;
    # scales from tests_ongoing's initialize_geodesic.yaml.
    stage_repel = {
        "potential_params": {"name": "repel"},
        "integrator_params": {
            **integrator_params,
            "path_integrand_names": ["geodesic"],
        },
        "optimizer_params": OPTIMIZER_PARAMS,
        "num_optimizer_iterations": N_ITERATIONS,
    }
    stage_morse = {
        "potential_params": {"name": "morse"},
        "integrator_params": {
            **integrator_params,
            "path_integrand_names": ["pvre", "geodesic"],
            "path_integrand_scales": [1.0, 0.05],
        },
        "optimizer_params": OPTIMIZER_PARAMS,
        "num_optimizer_iterations": N_ITERATIONS,
    }
    reference = read(XYZ, index=":")
    n_atoms = len(reference[0])

    # ---- Part A: one call per stage, so the path can be measured between them.
    mep = Popcornn(**POPCORNN_KWARGS)
    params_initial = {
        name: p.detach().clone() for name, p in mep.path.named_parameters()
    }
    d_min_initial = _min_interatomic_distance(mep.get_discrete_path())

    converged = mep.optimize_path(stage_repel)
    # [1] No threshold, so the stage hits its iteration cap.
    assert converged == [False], f"stage 1 returned {converged}"
    assert type(converged[0]) is bool
    # [2] Every parameter tensor moved, so the gradient reached every layer.
    params_after_repel = {
        name: p.detach().clone() for name, p in mep.path.named_parameters()
    }
    assert params_after_repel.keys() == params_initial.keys()
    for name, before in params_initial.items():
        assert not torch.equal(before, params_after_repel[name]), (
            f"stage 1 did not change {name}"
        )
    # [Behavior] The repel stage exists to pull clashing atoms apart.
    # Measured over starting meshes 0-9: 0.3946 Å rises by 0.0093 Å. A sign
    # error lowers it to 0.3854 Å, and a dead stage leaves it unchanged. No
    # margin: the claim is only that atoms separate, and the rise's size
    # depends on lr and N_ITERATIONS, while round-off is about 1e-15 Å.
    d_min_after_repel = _min_interatomic_distance(mep.get_discrete_path())
    assert d_min_after_repel > d_min_initial, (
        f"repel stage did not separate atoms: {d_min_initial:.4f} -> {d_min_after_repel:.4f} Å"
    )

    converged = mep.optimize_path(stage_morse)
    assert converged == [False], f"stage 2 returned {converged}"
    # [2] Stage 2 trained the parameters stage 1 left, with Morse attached.
    assert isinstance(mep.path.potential, MorsePotential)
    params_split = {name: p.detach().clone() for name, p in mep.path.named_parameters()}
    for name, after in params_split.items():
        assert not torch.equal(params_after_repel[name], after), (
            f"stage 2 did not change {name}"
        )

    # [4] ASE path output. Check the length first so an empty list fails clearly.
    frames = mep.get_discrete_path()
    assert len(frames) == POPCORNN_KWARGS["num_record_points"]
    for i, frame in enumerate(frames):
        assert isinstance(frame, Atoms)
        assert frame.get_chemical_symbols() == reference[0].get_chemical_symbols(), (
            f"frame {i}"
        )
        # [6]
        assert np.isfinite(frame.positions).all(), f"frame {i} has non-finite positions"
    # pbc, cell and tags are not checked: hard.xyz uses ASE's defaults for all
    # three, so they cannot catch an error. The frames carry no energies by
    # design; nothing in popcornn reads them (#96).

    # [3] Endpoints are pinned by the (1 - t) t factor in MLPpath.get_positions.
    # The only error is rounding in initial + t (final - initial) at t = 1:
    # worst 2.2e-16 Å over starting meshes 0-9, so the limit is 5e-15 Å
    # (smallest {1,2,5}x10^k >= 10x the worst).
    np.testing.assert_allclose(
        frames[0].positions, reference[0].positions, rtol=0, atol=5e-15
    )
    np.testing.assert_allclose(
        frames[-1].positions, reference[-1].positions, rtol=0, atol=5e-15
    )

    # [5] TS frame. Its energy must equal stage 2's (Morse) TS energy, which
    # shows get_ts used the TS from the last stage. get_ts evaluates one point,
    # ts_search a batch of candidates, so the two can round differently: worst
    # 3.2e-16 relative over starting meshes 0-9, so the limit is 5e-15. A stale
    # TS from stage 1 differs by about 1e-2 relative.
    ts = mep.get_ts()
    assert isinstance(ts, Atoms)
    assert np.isfinite(ts.positions).all()
    assert ts.get_potential_energy() == pytest.approx(
        float(mep.path.ts_energy), rel=5e-15
    )
    assert ts.get_forces().shape == (n_atoms, 3)
    assert np.isfinite(ts.get_forces()).all()

    # ---- Part B: both stages in one call, as users run them (examples/run.py).
    mep_combined = Popcornn(**POPCORNN_KWARGS)
    converged = mep_combined.optimize_path(stage_repel, stage_morse)
    # [1] One flag per stage.
    assert converged == [False, False], f"one call returned {converged}"
    # [2] The stages share one path, and optimize_path carries no other state
    # from one stage to the next, so one call gives exactly the parameters of
    # two calls (measured bit-identical on CPU).
    params_combined = {
        name: p.detach() for name, p in mep_combined.path.named_parameters()
    }
    assert params_combined.keys() == params_split.keys()
    for name, p in params_combined.items():
        assert torch.equal(p, params_split[name]), (
            f"{name} differs between one call and two calls"
        )


def test_output_frames_keep_fixed_atoms(tmp_path):
    """Output frames carry the input's FixAtoms indices, one entry per atom,
    and can be written (a per-coordinate mask gave indices up to 3N - 1, which
    made writing a frame raise IndexError)."""
    frames = read(XYZ, index=":")
    for atoms in frames:
        atoms.set_constraint(FixAtoms(indices=[16]))
    mep = Popcornn(
        images=frames,
        path_params={"name": "mlp", "width": 8, "depth": 2},
        device="cpu",
        dtype="float64",
    )
    output = mep.get_discrete_path(num_record_points=3)
    for atoms in output:
        assert list(atoms.constraints[0].index) == [16]
    write(tmp_path / "path.xyz", output, format="extxyz")
