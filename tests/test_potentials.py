from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pytest
import torch
from ase.calculators.calculator import Calculator
from ase.calculators.lj import LennardJones
from ase.calculators.morse import MorsePotential
from ase.data import covalent_radii
from ase.io import read
from ase.mep import interpolate
from ase.neighborlist import NeighborList
from fairchem.core import FAIRChemCalculator, pretrained_mlip

from popcornn.paths import get_path
from popcornn.potentials import get_potential
from popcornn.tools import process_images
from tests import IMAGES_DIR, REFERENCE_TOL

if TYPE_CHECKING:
    from ase import Atoms


def test_sphere(dtype, device):
    images = process_images(str(IMAGES_DIR / "sphere.json"), device=device, dtype=dtype)
    path = get_path("linear", images=images, device=device, dtype=dtype)
    potential = get_potential("sphere", images=images, device=device, dtype=dtype)

    potential_output = potential(
        path(
            torch.tensor(
                [0.0, 0.5, 1.0], requires_grad=True, device=device, dtype=dtype
            )
        ).positions
    )
    assert potential_output.energies.shape == (3, 1)
    assert potential_output.energies.device.type == device.type
    assert potential_output.energies.dtype == dtype
    assert torch.allclose(
        potential_output.energies,
        torch.tensor([[2.0], [1.0], [2.0]], device=device, dtype=dtype),
        atol=1e-5,
    )
    assert potential_output.energies.grad_fn is not None
    assert potential_output.energies_decomposed is None
    assert potential_output.forces.shape == (3, 2)
    assert potential_output.forces.device.type == device.type
    assert potential_output.forces.dtype == dtype
    assert torch.allclose(
        potential_output.forces,
        torch.tensor(
            [[2.0, -2.0], [0.0, -2.0], [-2.0, -2.0]], device=device, dtype=dtype
        ),
        atol=1e-5,
    )
    assert potential_output.forces.grad_fn is not None
    assert potential_output.forces_decomposed is None


def test_muller_brown(dtype, device):
    images = process_images(
        str(IMAGES_DIR / "muller_brown.json"), device=device, dtype=dtype
    )
    path = get_path("linear", images=images, device=device, dtype=dtype)
    potential = get_potential("muller_brown", images=images, device=device, dtype=dtype)

    potential_output = potential(
        path(
            torch.tensor(
                [0.0, 0.5, 1.0], requires_grad=True, device=device, dtype=dtype
            )
        ).positions
    )
    assert potential_output.energies.shape == (3, 1)
    assert potential_output.energies.device.type == device.type
    assert potential_output.energies.dtype == dtype
    assert torch.allclose(
        potential_output.energies,
        torch.tensor(
            [[-146.69948920058778], [3.225535000493114], [-80.76774924875772]],
            device=device,
            dtype=dtype,
        ),
        atol=1e-5,
    )
    assert potential_output.energies.grad_fn is not None
    assert potential_output.energies_decomposed is None
    assert potential_output.forces.shape == (3, 2)
    assert potential_output.forces.device.type == device.type
    assert potential_output.forces.dtype == dtype
    assert torch.allclose(
        potential_output.forces,
        torch.tensor(
            [
                [0.0001873538288492682, -0.20449389471367851],
                [-63.037220140249005, -57.82076869131242],
                [-0.048943927961211386, -0.44871341799094555],
            ],
            device=device,
            dtype=dtype,
        ),
        atol=1e-3,
    )
    assert potential_output.forces.grad_fn is not None
    assert potential_output.forces_decomposed is None


def test_wolfe(dtype, device):
    images = process_images(str(IMAGES_DIR / "wolfe.json"), device=device, dtype=dtype)
    path = get_path("linear", images=images, device=device, dtype=dtype)
    potential = get_potential(
        "wolfe_schlegel", images=images, device=device, dtype=dtype
    )

    potential_output = potential(
        path(
            torch.tensor(
                [0.0, 0.5, 1.0], requires_grad=True, device=device, dtype=dtype
            )
        ).positions
    )
    assert potential_output.energies.shape == (3, 1)
    assert potential_output.energies.device.type == device.type
    assert potential_output.energies.dtype == dtype
    assert torch.allclose(
        potential_output.energies,
        torch.tensor(
            [[-64.81812976863], [-0.04301175469875001], [-66.45448705023]],
            device=device,
            dtype=dtype,
        ),
        atol=1e-5,
    )
    assert potential_output.energies.grad_fn is not None
    assert potential_output.energies_decomposed is None
    assert potential_output.forces.shape == (3, 2)
    assert potential_output.forces.device.type == device.type
    assert potential_output.forces.dtype == dtype
    assert torch.allclose(
        potential_output.forces,
        torch.tensor(
            [
                [0.0032145200000005536, 0.04517024000000619],
                [-2.614820315, -1.194996355],
                [-0.0003081600000115481, -0.06473332000001358],
            ],
            device=device,
            dtype=dtype,
        ),
        atol=1e-3,
    )
    assert potential_output.forces.grad_fn is not None
    assert potential_output.forces_decomposed is None


def test_lennard_jones(dtype, device):
    raw_images = [read(str(IMAGES_DIR / "LJ13.xyz"), index=i) for i in (0, 1, 1)]
    interpolate(raw_images)
    for image in raw_images:
        image.calc = LennardJones()
    images = process_images(str(IMAGES_DIR / "LJ13.xyz"), device=device, dtype=dtype)
    path = get_path("linear", images=images, device=device, dtype=dtype)
    potential = get_potential(
        "lennard_jones", cutoff=3.0, images=images, device=device, dtype=dtype
    )
    potential_output = potential(
        path(
            torch.tensor(
                [0.0, 0.5, 1.0], requires_grad=True, device=device, dtype=dtype
            )
        ).positions
    )
    assert potential_output.energies.shape == (3, 1)
    assert potential_output.energies.device.type == device.type
    assert potential_output.energies.dtype == dtype
    assert torch.allclose(
        potential_output.energies,
        torch.tensor(
            [[image.get_potential_energy()] for image in raw_images],
            device=device,
            dtype=dtype,
        ),
        atol=1e-5,
    )
    assert potential_output.energies.grad_fn is not None
    assert potential_output.energies_decomposed.shape == (3, 78)
    assert potential_output.energies_decomposed.device.type == device.type
    assert potential_output.energies_decomposed.dtype == dtype
    assert torch.allclose(
        potential_output.energies_decomposed.sum(dim=-1, keepdim=True),
        potential_output.energies,
        atol=1e-5,
    )
    assert potential_output.energies_decomposed.grad_fn is not None
    assert potential_output.forces.shape == (3, 39)
    assert potential_output.forces.device.type == device.type
    assert potential_output.forces.dtype == dtype
    assert torch.allclose(
        potential_output.forces,
        torch.tensor(
            [image.get_forces().flatten() for image in raw_images],
            device=device,
            dtype=dtype,
        ),
        atol=1e-3,
    )
    assert potential_output.forces.grad_fn is not None
    assert potential_output.forces_decomposed.shape == (3, 78, 39)
    assert potential_output.forces_decomposed.device.type == device.type
    assert potential_output.forces_decomposed.dtype == dtype
    assert torch.allclose(
        potential_output.forces_decomposed.sum(dim=-2, keepdim=False),
        potential_output.forces,
        atol=1e-5,
    )
    assert potential_output.forces_decomposed.grad_fn is not None

    pytest.skip("Switching the full radius graph to minimal image convention")
    raw_images = [read(str(IMAGES_DIR / "LJ35.xyz"), index=i) for i in (0, 1, 1)]
    interpolate(raw_images, mic=True)
    for image in raw_images:
        image.calc = LennardJones()
    images = process_images(str(IMAGES_DIR / "LJ35.xyz"), device=device, dtype=dtype)
    path = get_path("linear", images=images, device=device, dtype=dtype)
    potential = get_potential(
        "lennard_jones", cutoff=3.0, images=images, device=device, dtype=dtype
    )
    potential_output = potential(
        path(
            torch.tensor(
                [0.0, 0.5, 1.0], requires_grad=True, device=device, dtype=dtype
            )
        ).positions
    )
    assert potential_output.energies.shape == (3, 1)
    assert potential_output.energies.device.type == device.type
    assert potential_output.energies.dtype == dtype
    assert torch.allclose(
        potential_output.energies,
        torch.tensor(
            [[image.get_potential_energy()] for image in raw_images],
            device=device,
            dtype=dtype,
        ),
        atol=1e-5,
    )
    assert potential_output.energies.grad_fn is not None
    assert potential_output.energies_decomposed.shape == (3, 4352)
    assert potential_output.energies_decomposed.device.type == device.type
    assert potential_output.energies_decomposed.dtype == dtype
    assert torch.allclose(
        potential_output.energies_decomposed.sum(dim=-1, keepdim=True),
        potential_output.energies,
        atol=1e-5,
    )
    assert potential_output.energies_decomposed.grad_fn is not None
    assert potential_output.forces.shape == (3, 105)
    assert potential_output.forces.device.type == device.type
    assert potential_output.forces.dtype == dtype
    assert torch.allclose(
        potential_output.forces,
        torch.tensor(
            [image.get_forces().flatten() for image in raw_images],
            device=device,
            dtype=dtype,
        ),
        atol=1e-3,
    )
    assert potential_output.forces.grad_fn is not None
    assert potential_output.forces_decomposed.shape == (3, 4352, 105)
    assert potential_output.forces_decomposed.device.type == device.type
    assert potential_output.forces_decomposed.dtype == dtype
    assert torch.allclose(
        potential_output.forces_decomposed.sum(dim=-2, keepdim=False),
        potential_output.forces,
        atol=1e-5,
    )
    assert potential_output.forces_decomposed.grad_fn is not None


def test_repel(dtype, device):
    class RepelCalculator(Calculator):
        """
        Revised from ASE LennardJones Calculator to use the repulsive potential
        """

        implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

        def __init__(self, alpha=1.7, beta=0.01, cutoff=3.0, **kwargs):
            super().__init__(**kwargs)
            self.alpha = alpha
            self.beta = beta
            self.rc = cutoff
            self.nl = None

        def calculate(self, atoms=None, properties=None, system_changes=None):
            Calculator.calculate(self, atoms, properties, system_changes)

            natoms = len(self.atoms)
            alpha = self.alpha
            beta = self.beta
            rc = self.rc
            if self.nl is None or "numbers" in system_changes:
                self.nl = NeighborList(
                    [rc / 2] * natoms, self_interaction=False, bothways=True
                )
            self.nl.update(self.atoms)
            positions = self.atoms.positions
            cell = self.atoms.cell

            energies = np.zeros(natoms)
            forces = np.zeros((natoms, 3))
            for ii in range(natoms):
                neighbors, offsets = self.nl.get_neighbors(ii)
                cells = np.dot(offsets, cell)
                distance_vectors = positions[neighbors] + cells - positions[ii]
                r = np.linalg.norm(distance_vectors, axis=1)
                r0 = (
                    covalent_radii[self.atoms.numbers[ii]]
                    + covalent_radii[self.atoms.numbers[neighbors]]
                )
                pairwise_energies = np.exp(-alpha * (r - r0) / r0) + beta * r0 / r
                pairwise_energies[r > rc] = 0.0
                pairwise_forces = (
                    -np.exp(-alpha * (r - r0) / r0) * alpha / r0 / r - beta * r0 / r**3
                )
                pairwise_forces[r > rc] = 0.0
                pairwise_energies -= (
                    np.exp(-alpha * (rc - r0) / r0) + beta * r0 / rc
                ) * (r < rc)
                pairwise_forces = pairwise_forces[:, np.newaxis] * distance_vectors
                energies[ii] += 0.5 * pairwise_energies.sum()
                forces[ii] += pairwise_forces.sum(axis=0)
            energy = energies.sum()
            self.results["energy"] = energy
            self.results["forces"] = forces

    raw_images = [read(str(IMAGES_DIR / "LJ13.xyz"), index=i) for i in (0, 1, 1)]
    interpolate(raw_images)
    for image in raw_images:
        image.calc = RepelCalculator()
    images = process_images(str(IMAGES_DIR / "LJ13.xyz"), device=device, dtype=dtype)
    path = get_path("linear", images=images, device=device, dtype=dtype)
    potential = get_potential(
        "repel", cutoff=3.0, images=images, device=device, dtype=dtype
    )
    potential_output = potential(
        path(
            torch.tensor(
                [0.0, 0.5, 1.0], requires_grad=True, device=device, dtype=dtype
            )
        ).positions
    )
    assert potential_output.energies.shape == (3, 1)
    assert potential_output.energies.device.type == device.type
    assert potential_output.energies.dtype == dtype
    assert torch.allclose(
        potential_output.energies,
        torch.tensor(
            [[image.get_potential_energy()] for image in raw_images],
            device=device,
            dtype=dtype,
        ),
        atol=1e-5,
    )
    assert potential_output.energies.grad_fn is not None
    assert potential_output.energies_decomposed.shape == (3, 78)
    assert potential_output.energies_decomposed.device.type == device.type
    assert potential_output.energies_decomposed.dtype == dtype
    assert torch.allclose(
        potential_output.energies_decomposed.sum(dim=-1, keepdim=True),
        potential_output.energies,
        atol=1e-5,
    )
    assert potential_output.energies_decomposed.grad_fn is not None
    assert potential_output.forces.shape == (3, 39)
    assert potential_output.forces.device.type == device.type
    assert potential_output.forces.dtype == dtype
    assert torch.allclose(
        potential_output.forces,
        torch.tensor(
            [image.get_forces().flatten() for image in raw_images],
            device=device,
            dtype=dtype,
        ),
        atol=1e-3,
    )
    assert potential_output.forces.grad_fn is not None
    assert potential_output.forces_decomposed.shape == (3, 78, 39)
    assert potential_output.forces_decomposed.device.type == device.type
    assert potential_output.forces_decomposed.dtype == dtype
    assert torch.allclose(
        potential_output.forces_decomposed.sum(dim=-2, keepdim=False),
        potential_output.forces,
        atol=1e-5,
    )
    assert potential_output.forces_decomposed.grad_fn is not None

    pytest.skip("Switching the full radius graph to minimal image convention")
    raw_images = [read(str(IMAGES_DIR / "LJ35.xyz"), index=i) for i in (0, 1, 1)]
    interpolate(raw_images, mic=True)
    for image in raw_images:
        image.calc = RepelCalculator()
    images = process_images(str(IMAGES_DIR / "LJ35.xyz"), device=device, dtype=dtype)
    path = get_path("linear", images=images, device=device, dtype=dtype)
    potential = get_potential(
        "repel", cutoff=3.0, images=images, device=device, dtype=dtype
    )
    potential_output = potential(
        path(
            torch.tensor(
                [0.0, 0.5, 1.0], requires_grad=True, device=device, dtype=dtype
            )
        ).positions
    )
    assert potential_output.energies.shape == (3, 1)
    assert potential_output.energies.device.type == device.type
    assert potential_output.energies.dtype == dtype
    assert torch.allclose(
        potential_output.energies,
        torch.tensor(
            [[image.get_potential_energy()] for image in raw_images],
            device=device,
            dtype=dtype,
        ),
        atol=1e-5,
    )
    assert potential_output.energies.grad_fn is not None
    assert potential_output.energies_decomposed.shape == (3, 4352)
    assert potential_output.energies_decomposed.device.type == device.type
    assert potential_output.energies_decomposed.dtype == dtype
    assert torch.allclose(
        potential_output.energies_decomposed.sum(dim=-1, keepdim=True),
        potential_output.energies,
        atol=1e-5,
    )
    assert potential_output.energies_decomposed.grad_fn is not None
    assert potential_output.forces.shape == (3, 105)
    assert potential_output.forces.device.type == device.type
    assert potential_output.forces.dtype == dtype
    assert torch.allclose(
        potential_output.forces,
        torch.tensor(
            [image.get_forces().flatten() for image in raw_images],
            device=device,
            dtype=dtype,
        ),
        atol=1e-3,
    )
    assert potential_output.forces.grad_fn is not None
    assert potential_output.forces_decomposed.shape == (3, 4352, 105)
    assert potential_output.forces_decomposed.device.type == device.type
    assert potential_output.forces_decomposed.dtype == dtype
    assert torch.allclose(
        potential_output.forces_decomposed.sum(dim=-2, keepdim=False),
        potential_output.forces,
        atol=1e-5,
    )
    assert potential_output.forces_decomposed.grad_fn is not None


def pair_reference(
    atoms: Atoms, name: str, params: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pair energies and forces of popcornn's ``morse`` and ``harmonic``
    potentials, by an explicit loop over atom pairs.

    Pair energies, with :math:`r_0 = R^{cov}_i + R^{cov}_j` the sum of covalent
    radii from ``ase.data.covalent_radii`` (Cordero et al., 2008, Dalton Trans.,
    2832-2838, https://doi.org/10.1039/B801115J):

    - morse: :math:`E_{ij} = (1 - e^{-\\alpha (r_{ij} - r_0)})^2 - 1`, the Morse
      potential (P. M. Morse, 1929, Phys. Rev. 34, 57,
      https://doi.org/10.1103/PhysRev.34.57) with unit well depth, shifted by
      -1 so that the energy vanishes at large separation;
    - harmonic: :math:`E_{ij} = (r_{ij} - r_0)^2 \\,\\sigma((r_{max} - r_{ij}) / skin)`,
      with :math:`\\sigma` the logistic sigmoid.

    Forces use the hand-derived :math:`dE/dr` (independent of autograd):
    morse :math:`2(1 - x)\\alpha x` with :math:`x = e^{-\\alpha (r - r_0)}`;
    harmonic :math:`2(r - r_0)\\sigma + (r - r_0)^2 \\sigma'` with
    :math:`\\sigma' = -\\sigma(1 - \\sigma) / skin`.

    Plain Cartesian distances: non-periodic systems only.

    Parameters
    ----------
    atoms : Atoms
        A non-periodic structure.
    name : str
        ``"morse"`` or ``"harmonic"``.
    params : dict[str, float]
        Exactly ``{"alpha"}`` for morse, or ``{"r_max", "skin"}`` for harmonic.
        Every key is required, so the expected values are pinned by the caller.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Per-pair energies, shape ``(n_pairs,)``, and per-pair forces, shape
        ``(n_pairs, 3 * n_atoms)``, with pairs ordered i < j, i outer (the
        order of ``torch.triu_indices(n, n, offset=1)``).
    """
    required = {"morse": {"alpha"}, "harmonic": {"r_max", "skin"}}
    if atoms.pbc.any():
        msg = "pair_reference uses plain Cartesian distances; got a periodic system."
        raise ValueError(msg)
    if name not in required:
        msg = f"Unknown pair potential {name!r}; expected one of {sorted(required)}."
        raise ValueError(msg)
    if set(params) != required[name]:
        msg = f"{name} needs exactly {sorted(required[name])}, got {sorted(params)}."
        raise ValueError(msg)

    positions = atoms.positions
    n_atoms = len(atoms)
    pair_energies, pair_forces = [], []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            r0 = covalent_radii[atoms.numbers[i]] + covalent_radii[atoms.numbers[j]]
            d = positions[i] - positions[j]
            r = np.linalg.norm(d)
            if name == "morse":
                x = np.exp(-params["alpha"] * (r - r0))
                energy = (1 - x) ** 2 - 1
                de_dr = 2 * (1 - x) * params["alpha"] * x
            elif name == "harmonic":
                s = 1 / (1 + np.exp(-(params["r_max"] - r) / params["skin"]))
                energy = (r - r0) ** 2 * s
                de_dr = 2 * (r - r0) * s - (r - r0) ** 2 * s * (1 - s) / params["skin"]
            force = np.zeros((n_atoms, 3))
            force[i] = -de_dr * d / r
            force[j] = de_dr * d / r
            pair_energies.append(energy)
            pair_forces.append(force.ravel())
    return np.array(pair_energies), np.array(pair_forces)


@pytest.mark.parametrize("system", ["LJ13.xyz", "T1x.xyz"])
@pytest.mark.parametrize(
    ("name", "params", "reference_params"),
    [
        # `params` goes to popcornn; `reference_params` pins the values the reference
        # must use. The default cases pass {} to popcornn, so they also pin the
        # documented defaults.
        pytest.param("morse", {}, {"alpha": 2.0}, id="morse-default"),
        pytest.param("morse", {"alpha": 1.3}, {"alpha": 1.3}, id="morse-alpha1.3"),
        pytest.param(
            "harmonic", {}, {"r_max": 1.5, "skin": 0.1}, id="harmonic-default"
        ),
        pytest.param(
            "harmonic",
            {"r_max": 2.0, "skin": 0.2},
            {"r_max": 2.0, "skin": 0.2},
            id="harmonic-rmax2-skin0.2",
        ),
    ],
)
def test_pair_potentials(name, params, reference_params, system, dtype, device):
    """Morse and harmonic energies and forces, totals and per pair, against
    ``pair_reference``; Morse totals also against ASE where it applies.

    T1x (C3H8O2) is needed besides LJ13: all LJ13 pairs share one r0, so a
    mismatch between pairs and their r0 would pass on LJ13 alone.
    """
    data = str(IMAGES_DIR / system)
    raw_images = [read(data, index=i) for i in (0, 1, 1)]
    interpolate(raw_images)
    references = [pair_reference(image, name, reference_params) for image in raw_images]
    energies_decomposed_ref = torch.tensor(
        np.stack([e for e, _ in references]), device=device, dtype=dtype
    )
    forces_decomposed_ref = torch.tensor(
        np.stack([f for _, f in references]), device=device, dtype=dtype
    )

    images = process_images(data, device=device, dtype=dtype)
    path = get_path("linear", images=images, device=device, dtype=dtype)
    potential = get_potential(name, images=images, device=device, dtype=dtype, **params)
    output = potential(
        path(
            torch.tensor(
                [0.0, 0.5, 1.0], requires_grad=True, device=device, dtype=dtype
            )
        ).positions
    )

    n_atoms = len(raw_images[0])
    n_pairs = n_atoms * (n_atoms - 1) // 2
    expected = {
        "energies": energies_decomposed_ref.sum(dim=-1, keepdim=True),
        "energies_decomposed": energies_decomposed_ref,
        "forces": forces_decomposed_ref.sum(dim=-2),
        "forces_decomposed": forces_decomposed_ref,
    }
    expected_shapes = {
        "energies": (3, 1),
        "energies_decomposed": (3, n_pairs),
        "forces": (3, 3 * n_atoms),
        "forces_decomposed": (3, n_pairs, 3 * n_atoms),
    }
    for field, reference in expected.items():
        tensor = getattr(output, field)
        assert tensor.shape == expected_shapes[field], field
        assert tensor.device.type == device.type, field
        assert tensor.dtype == dtype, field
        assert tensor.grad_fn is not None, field
        torch.testing.assert_close(
            tensor,
            reference,
            rtol=0.0,
            atol=REFERENCE_TOL[dtype] * reference.abs().max().item(),
            msg=lambda m, field=field: f"{field}: {m}",
        )

    # Ties the Morse formula to third-party code (ASE), where ASE applies: one element.
    # ASE's pair energy eps*e^{rho0(1-d/r0)}(e^{rho0(1-d/r0)}-2) equals the Morse E_ij
    # above when eps=1 and rho0=alpha*r0. rcut1 and rcut2 are in units of r0 and are
    # set out of range, so ASE's smooth cutoff never acts.
    if name == "morse" and len(set(raw_images[0].numbers)) == 1:
        r0 = 2 * covalent_radii[raw_images[0].numbers[0]]
        ase_morse = MorsePotential(
            epsilon=1.0,
            rho0=reference_params["alpha"] * r0,
            r0=r0,
            rcut1=100.0,
            rcut2=101.0,
        )
        ase_energies, ase_forces = [], []
        for image in raw_images:
            image.calc = ase_morse
            ase_energies.append([image.get_potential_energy()])
            ase_forces.append(image.get_forces().flatten())
        for tensor, ase_values in (
            (output.energies, ase_energies),
            (output.forces, ase_forces),
        ):
            reference = torch.tensor(np.array(ase_values), device=device, dtype=dtype)
            torch.testing.assert_close(
                tensor,
                reference,
                rtol=0.0,
                atol=REFERENCE_TOL[dtype] * reference.abs().max().item(),
            )


@pytest.mark.parametrize(
    "device", [torch.device("cpu"), torch.device("cuda"), torch.device("cuda", 1)]
)
def test_uma(dtype, device):
    pytest.skip(reason="Give github an UMA license")
    if device.type == "cuda" and torch.cuda.device_count() <= (device.index or 0):
        pytest.skip(reason=f"{device} is not available, skipping test.")

    # if dtype == torch.float64:
    #     pytest.skip("UMA potential is not supported for float64 due to precision issues.")

    raw_images = [read(str(IMAGES_DIR / "T1x.xyz"), index=i) for i in (0, 1, 1)]
    interpolate(raw_images)
    for image in raw_images:
        image.calc = FAIRChemCalculator(
            pretrained_mlip.get_predict_unit("uma-s-1", device=device.type),
            task_name="omol",
        )
    images = process_images(str(IMAGES_DIR / "T1x.xyz"), device=device, dtype=dtype)
    path = get_path("linear", images=images, device=device, dtype=dtype)
    potential = get_potential(
        "uma",
        model_name="uma-s-1",
        task_name="omol",
        images=images,
        device=device,
        dtype=dtype,
    )
    t = torch.tensor([0.0, 0.5, 1.0], requires_grad=True, device=device, dtype=dtype)
    positions = path(t).positions
    potential_output = potential(positions)
    assert potential_output.energies.shape == (3, 1)
    assert potential_output.energies.device == positions.device
    assert potential_output.energies.dtype == dtype
    assert torch.allclose(
        potential_output.energies,
        torch.tensor(
            [[image.get_potential_energy()] for image in raw_images],
            device=device,
            dtype=dtype,
        ),
        atol=1e-5,
    )
    assert potential_output.energies.grad_fn is not None
    assert potential_output.energies_decomposed is None
    assert potential_output.forces.shape == (3, 39)
    assert potential_output.forces.device == positions.device
    assert potential_output.forces.dtype == dtype
    assert torch.allclose(
        potential_output.forces,
        torch.tensor(
            [image.get_forces().flatten() for image in raw_images],
            device=device,
            dtype=dtype,
        ),
        atol=1e-3 if device.type == "cpu" else 1e-2,
    )
    assert potential_output.forces.grad_fn is not None
    assert potential_output.forces_decomposed is None
    (grad_t,) = torch.autograd.grad(potential_output.energies.sum(), t)
    assert grad_t.device == positions.device

    raw_images = [read(str(IMAGES_DIR / "OC20NEB.xyz"), index=i) for i in (0, 1, 1)]
    interpolate(raw_images)
    for image in raw_images:
        image.calc = FAIRChemCalculator(
            pretrained_mlip.get_predict_unit("uma-s-1", device=device.type),
            task_name="oc20",
        )
    images = process_images(str(IMAGES_DIR / "OC20NEB.xyz"), device=device, dtype=dtype)
    path = get_path("linear", images=images, device=device, dtype=dtype)
    potential = get_potential(
        "uma",
        model_name="uma-s-1",
        task_name="oc20",
        images=images,
        device=device,
        dtype=dtype,
    )
    t = torch.tensor([0.0, 0.5, 1.0], requires_grad=True, device=device, dtype=dtype)
    positions = path(t).positions
    potential_output = potential(positions)
    assert potential_output.energies.shape == (3, 1)
    assert potential_output.energies.device == positions.device
    assert potential_output.energies.dtype == dtype
    assert torch.allclose(
        potential_output.energies,
        torch.tensor(
            [[image.get_potential_energy()] for image in raw_images],
            device=device,
            dtype=dtype,
        ),
        atol=1e-5,
    )
    assert potential_output.energies.grad_fn is not None
    assert potential_output.energies_decomposed is None
    assert potential_output.forces.shape == (3, 42)
    assert potential_output.forces.device == positions.device
    assert potential_output.forces.dtype == dtype
    assert torch.allclose(
        potential_output.forces * ~images.fix_positions,
        torch.tensor(
            [image.get_forces().flatten() for image in raw_images],
            device=device,
            dtype=dtype,
        ),
        atol=1e-3 if device.type == "cpu" else 1e-2,
    )
    assert potential_output.forces.grad_fn is not None
    assert potential_output.forces_decomposed is None
