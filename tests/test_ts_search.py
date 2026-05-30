import torch
import numpy as np
from popcornn.tools.images import Images
from popcornn.tools import SamplesCache
from popcornn.potentials.base_potential import BasePotential, PotentialOutput
from popcornn.paths.base_path import BasePath


class LegendrePotential(BasePotential):
    """1-D Legendre-like potential over ``sum(x)``.

    Used to drive the TS-search test through a real ``BasePath.forward``
    (ts_search now always re-evaluates at the interpolated t).
    """

    def __init__(self, images, device, dtype, n):
        super().__init__(images, device=device, dtype=dtype)
        self.n = n

    def forward(self, positions):
        positions = positions.requires_grad_(True)
        s = positions.sum(dim=-1, keepdim=True)
        if self.n == 2:
            E = -1 * (3 * s ** 2 - 1) / 2.
        elif self.n == 3:
            E = (5 * s ** 3 - 2 * s ** 2 - 3 * s) / 2.
        elif self.n == 4:
            E = (35 * s ** 4 - 30 * s ** 2 + 3) / 8. - 0.75 * s ** 2
        else:
            raise ValueError(self.n)
        F = self.calculate_conservative_forces(E, positions, create_graph=False)
        return PotentialOutput(energies=E, forces=F)


def test_ts_search():
    # Setup environment
    torch.manual_seed(2025)
    np.random.seed(2025)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float64

    # Create initial/end points
    x_init = torch.tensor([-1.5, 1, -0.5], device=device, dtype=dtype)
    x_final = torch.tensor([-0.25, 0.25, 1], device=device, dtype=dtype)
    images = Images(
        image_type=dtype,
        positions=torch.stack([x_init, x_final]),
        fix_positions=None,
    )

    def path(t):
        # Linear interpolation; velocity = x_final - x_init (constant in t).
        return x_init + (x_final - x_init) * t

    truth = {
        2: (0.5, 0.5),
        3: (0.5 + (-1./3) / 2., 8./27),
        4: (0.5, 0.375),
    }

    for l in [2, 3, 4]:
        # Build a fresh BasePath + LegendrePotential per l so the fresh
        # forward inside ts_search has a working potential.
        base_path = BasePath(images=images, dtype=dtype, device=device)
        potential = LegendrePotential(images, device=device, dtype=dtype, n=l)
        base_path.set_potential(potential)
        # Override get_positions so the linear-interp `path(t)` drives
        # the fresh forward as well as the sample construction.
        base_path.get_positions = lambda t: path(t)
        # calculate_velocities uses autograd-jacobian over get_positions,
        # which works on the lambda above.

        # Build samples on a 13-pt mesh.
        time = torch.linspace(
            0, 1, 13, device=device, dtype=dtype, requires_grad=True
        ).unsqueeze(-1)
        positions = path(time)
        s = positions.sum(dim=-1, keepdim=True)
        if l == 2:
            E = -1 * (3 * s ** 2 - 1) / 2.
        elif l == 3:
            E = (5 * s ** 3 - 2 * s ** 2 - 3 * s) / 2.
        else:
            E = (35 * s ** 4 - 30 * s ** 2 + 3) / 8. - 0.75 * s ** 2
        F = BasePotential.calculate_conservative_forces(E, positions, create_graph=False)
        v = (x_final - x_init).expand_as(F)
        dEdt = -(F * v).sum(dim=-1)

        samples = SamplesCache(
            time=time[:, 0].detach(),
            energies=E.detach(),
            dEdt=dEdt.detach(),
        )
        base_path.ts_search(samples)

        ts_time_truth, ts_energy_truth = truth[l]
        assert np.isclose(
            ts_time_truth, base_path.ts_time.cpu().item(), atol=1e-2, rtol=1e-4
        ), (
            f"Did not match TS times for legendre {l}, got "
            f"{base_path.ts_time.item()}, expected {ts_time_truth}"
        )
        assert np.isclose(
            ts_energy_truth, base_path.ts_energy.flatten().cpu().item(),
            atol=1e-3,
        ), (
            f"Did not match TS energy for legendre {l}, got "
            f"{base_path.ts_energy.item()}, expected {ts_energy_truth}"
        )
        assert base_path.ts_force_mag.item() < 1e-3, (
            f"Did not find sufficiently small TS force magnitude for "
            f"legendre {l}, got {base_path.ts_force_mag.item()}"
        )
