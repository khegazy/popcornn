
import torch

from .base_potential import BasePotential, PotentialOutput

class WolfeSchlegel(BasePotential):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.minima = torch.tensor([[-1.166, 1.477], [-1.0, -1.5], [1.133, -1.486]])

    def forward(self, points):
        # points = self.point_transform(points)
        #points = torch.movedim(points, -1, 0)
        x = points[:,0]
        y = points[:,1]
        energy = 10*(x**4 + y**4 - 2*x**2 - 4*y**2\
            + x*y + 0.2*x + 0.1*y)
        energy = energy.unsqueeze(-1)
        force = self.calculate_conservative_force(energy, points)
        return PotentialOutput(
            energy=energy,
            force=force
        )