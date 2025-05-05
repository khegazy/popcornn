import torch
from .base_potential import BasePotential, PotentialOutput


class Schwefel(BasePotential):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def forward(self, positions):
        dim = positions.shape[-1]
        offset = 418.9829 * dim
        sinusiods = positions * torch.sin(torch.sqrt(torch.abs(positions)))
        energies_decomposed = offset - sinusiods
        energies = torch.sum(energies_decomposed, dim=-1, keepdim=True)
        forces = self.calculate_conservative_forces(energies, positions)
        forces_decomposed = self.calculate_conservative_forces_decomposed(energies_decomposed, positions)

        return PotentialOutput(
            energies=energies,
            energies_decomposed=energies_decomposed,
            forces=forces,
            forces_decomposed=forces_decomposed
        )