
import torch

from .base_potential import BasePotential, PotentialOutput

class WolfeSchlegel(BasePotential):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.minima = torch.tensor([[-1.166, 1.477], [-1.0, -1.5], [1.133, -1.486]])

    def forward(self, positions):
        x = positions[:,0]
        y = positions[:,1]
        energies = 10*(x**4 + y**4 - 2*x**2 - 4*y**2\
            + x*y + 0.2*x + 0.1*y)
        energies = energies.unsqueeze(-1)
        forces = self.calculate_conservative_forces(energies, positions)
        return PotentialOutput(
            energies=energies,
            forces=forces
        )