import torch
from .base_potential import BasePotential, PotentialOutput


class Schwefel(BasePotential):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def forward(self, points):
        dim = points.shape[-1]
        offset = 418.9829 * dim
        sinusiods = points * torch.sin(torch.sqrt(torch.abs(points)))
        energyterms = offset - sinusiods
        energy = torch.sum(energyterms, dim=-1, keepdim=True)
        force = self.calculate_conservative_force(energy, points)
        forceterms = self.calculate_conservative_forceterms(energyterms, points)

        return PotentialOutput(
            energy=energy,
            energyterms=energyterms,
            force=force,
            forceterms=forceterms
        )