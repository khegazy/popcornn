import torch
from ase.data import covalent_radii

from .base_potential import BasePotential, PotentialOutput
from popcornn.tools import wrap_positions

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
        v = positions_3d[:, self.ind[0]] - positions_3d[:, self.ind[1]]
        if self.pbc is not None and self.pbc.any():
            v = wrap_positions(v, self.cell, self.pbc, center=1.0)
        r = torch.norm(v, dim=-1)
        energies_decomposed = (
            torch.exp(-self.alpha * (r - self.r0) / self.r0) 
            + self.beta * self.r0 / r
        )
        energies = energies_decomposed.sum(dim=1, keepdim=True) 
        # forces_decomposed = self.calculate_conservative_forces_decomposed(energies_decomposed, positions)
        de_dr = (
            torch.exp(-self.alpha * (r - self.r0) / self.r0) * self.alpha / self.r0
            + self.beta * self.r0 / r ** 2
        )
        de_dv = de_dr[:, :, None] * v / r[:, :, None]
        forces_decomposed = torch.zeros(*energies_decomposed.shape, *positions_3d.shape[1:], device=positions.device, dtype=positions.dtype)
        forces_decomposed[:, torch.arange(self.ind.shape[1], device=self.device), self.ind[0], :] = de_dv
        forces_decomposed[:, torch.arange(self.ind.shape[1], device=self.device), self.ind[1], :] = -de_dv
        forces_decomposed = forces_decomposed.view(*energies_decomposed.shape, *positions.shape[1:])
        forces = forces_decomposed.sum(dim=1)
        return PotentialOutput(
            energies=energies,
            energies_decomposed=energies_decomposed,
            forces=forces,
            forces_decomposed=forces_decomposed,
        )

    def set_r0(self, atomic_numbers):
        """
        Set the r0_ij values for the potential
        """
        radii = torch.tensor([covalent_radii[n] for n in atomic_numbers], device=self.device)
        r0 = radii.view(-1, 1) + radii.view(1, -1)
        self.ind = torch.triu_indices(r0.shape[0], r0.shape[1], offset=1, device=self.device)
        # if self.fix_positions is not None:
        #     self.ind = self.ind[:, ~(self.fix_positions[self.ind[0]] & self.fix_positions[self.ind[1]])]
        self.r0 = r0[None, self.ind[0], self.ind[1]]

