import torch
from ase.data import covalent_radii

from .base_potential import BasePotential, PotentialOutput

class RepelPotential(BasePotential):
    def __init__(
            self, 
            alpha=1.7, 
            beta=0.01, 
            **kwargs,
        ):
        """
        Constructor for the Repulsive Potential from 
        Zhu, X., Thompson, K. C. & Martínez, T. J. 
        Geodesic interpolation for reaction pathways. 
        Journal of Chemical Physics 150, 164103 (2019).

        The potential is given by:
        E = sum_{i<j} exp(-alpha * (r_ij - r0_ij) / r0_ij) + beta * r0_ij / r_ij

        No cutoff is used in this implementation.

        Parameters
        ----------
        alpha: exponential term decay factor
        beta: inverse term weight
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.r0 = None
    
    def forward(self, positions):
        if self.r0 is None:
            self.set_r0(self.atomic_numbers)
            
        positions_3d = positions.view(-1, self.n_atoms, 3)
        r = torch.norm(positions_3d[:, self.ind[0]] - positions_3d[:, self.ind[1]], dim=-1)
        energies_decomposed = (
            (torch.exp(-self.alpha * (r - self.r0) / self.r0) + self.beta * self.r0 / r)
            # * torch.sigmoid((self.r_max - r) / self.skin)
        )
        energies = energies_decomposed.sum(dim=-1, keepdim=True) 
        return PotentialOutput(
            energies=energies,
            energies_decomposed=energies_decomposed,
            forces=self.calculate_conservative_forces(energies, positions),
            forces_decomposed=self.calculate_conservative_forces_decomposed(energies_decomposed, positions)
        )

    def set_r0(self, atomic_numbers):
        """
        Set the r0_ij values for the potential
        """
        radii = torch.tensor([covalent_radii[n] for n in atomic_numbers], device=self.device)
        r0 = radii.view(-1, 1) + radii.view(1, -1)
        self.ind = torch.triu_indices(r0.shape[0], r0.shape[1], offset=1, device=self.device)
        self.r0 = r0[None, self.ind[0], self.ind[1]]

