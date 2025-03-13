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
    
    def forward(self, points):
        if self.r0 is None:
            self.set_r0(self.numbers)
            
        points_3d = points.view(-1, self.n_atoms, 3)
        r = torch.norm(points_3d[:, self.ind[0]] - points_3d[:, self.ind[1]], dim=-1)
        energy_terms = (
            (torch.exp(-self.alpha * (r - self.r0) / self.r0) + self.beta * self.r0 / r)
            # * torch.sigmoid((self.r_max - r) / self.skin)
        )
        return PotentialOutput(energy_terms=energy_terms)
        # energy = energy_terms.sum(dim=-1, keepdim=True)
        # return PotentialOutput(energy=energy)

        # force = torch.vmap(
        #     lambda vec: torch.autograd.grad(
        #         energy_terms.flatten(), points, grad_outputs=vec, create_graph=True, retain_graph=True
        #     )[0],
        # )(torch.eye(energy_terms.shape[1], device=self.device).repeat(1, energy_terms.shape[0])).transpose(0, 1)
        # return PotentialOutput(energy=energy, force=force)
    
    def set_r0(self, numbers):
        """
        Set the r0_ij values for the potential
        """
        radii = torch.tensor([covalent_radii[n] for n in numbers], device=self.device)
        r0 = radii.view(-1, 1) + radii.view(1, -1)
        self.ind = torch.triu_indices(r0.shape[0], r0.shape[1], offset=1, device=self.device)
        self.r0 = r0[None, self.ind[0], self.ind[1]]

