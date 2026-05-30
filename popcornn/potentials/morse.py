import torch
from ase.data import covalent_radii

from .base_potential import BasePotential, PotentialOutput

class MorsePotential(BasePotential):
    """
    All-pairs Morse potential.

    Energy per pair:

    .. math::
        E_{ij} = \\big(1 - e^{-\\alpha (r_{ij} - r_0)}\\big)^2 - 1

    where ``r_0`` is the sum of covalent radii. Used as a cheap
    classical surface for testing and as the default for some
    relax-then-refine pipelines.
    """

    def __init__(self, alpha=2.0, **kwargs):
        """
        Parameters
        ----------
        alpha : float, default=2.0
            Morse range parameter (1/Å).
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.r0 = None
    
    def forward(self, positions):
        if self.r0 is None:
            self.set_r0(self.atomic_numbers)
        positions_3d = positions.view(-1, self.n_atoms, 3)
        r = torch.norm(positions_3d[:, self.ind[0]] - positions_3d[:, self.ind[1]], dim=-1)
        energies_decomposed = (1 - torch.exp(-self.alpha * (r - self.r0))) ** 2 - 1
        energies = torch.sum(energies_decomposed, dim=-1, keepdim=True)

        forces = self.calculate_conservative_forces(energies, positions)
        forces_decomposed = self.calculate_conservative_forces_decomposed(energies_decomposed, positions)
        return PotentialOutput(
            energies=energies,
            energies_decomposed=energies_decomposed,
            forces=forces,
            forces_decomposed=forces_decomposed
        )
    
    def set_r0(self, atomic_numbers):
        """
        Set the r0_ij values for the potential
        """
        radii = torch.tensor([covalent_radii[n] for n in atomic_numbers], device=self.device)
        r0 = radii.view(-1, 1) + radii.view(1, -1)
        self.ind = torch.triu_indices(r0.shape[0], r0.shape[1], offset=1, device=self.device)
        self.r0 = r0[None, self.ind[0], self.ind[1]]

