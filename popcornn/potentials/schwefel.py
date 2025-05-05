import torch
from .base_potential import BasePotential, PotentialOutput


class Schwefel(BasePotential):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def forward(self, positions):
        dim = positions.shape[-1]
        offset = 418.9829 * dim
        sinusiods = positions * torch.sin(torch.sqrt(torch.abs(positions)))
        energyterms = offset - sinusiods
        energies = torch.sum(energyterms, dim=-1, keepdim=True)
        forces = self.calculate_conservative_forces(energies, positions)
        forceterms = self.calculate_conservative_forceterms(energyterms, positions)

        return PotentialOutput(
            energies=energies,
            energyterms=energyterms,
            forces=forces,
            forceterms=forceterms
        )