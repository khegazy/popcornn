import torch
from ase.data import covalent_radii

from .base_potential import BasePotential, PotentialOutput

class HarmonicPotential(BasePotential):
    def __init__(self, r_max=1.5, skin=0.1, **kwargs):
        """
        Constructor for the Morse Potential

        The potential is given by:
        E_ij = (1 - exp(-alpha * (r_ij - r0))) ** 2 * sigmoid(beta * (sigma * r0 - r_ij))
        E = sum_{i<j} E_ij

        Parameters
        ----------
        """
        super().__init__(**kwargs)
        self.r_max = r_max
        self.skin = skin
        self.r0 = None
    
    def forward(self, points):
        if self.r0 is None:
            self.set_r0(self.numbers)
        points_3d = points.view(-1, self.n_atoms, 3)
        r = torch.norm(points_3d[:, self.ind[0]] - points_3d[:, self.ind[1]], dim=-1)
        energyterms = (r - self.r0) ** 2 * torch.sigmoid((self.r_max - r) / self.skin)
        energy = torch.sum(energyterms, dim=-1, keepdim=True)

        force = self.calculate_conservative_force(energy, points)
        forceterms = self.calculate_conservative_forceterms(energyterms, points)
        return PotentialOutput(
            energy=energy,
            energyterms=energyterms,
            force=force,
            forceterms=forceterms
        )
    
    def set_r0(self, numbers):
        """
        Set the r0_ij values for the potential
        """
        radii = torch.tensor([covalent_radii[n] for n in numbers], device=self.device)
        r0 = radii.view(-1, 1) + radii.view(1, -1)
        self.ind = torch.triu_indices(r0.shape[0], r0.shape[1], offset=1, device=self.device)
        self.r0 = r0[None, self.ind[0], self.ind[1]]

