from __future__ import annotations

import torch

from .base_potential import BasePotential, PotentialOutput


class Sphere(BasePotential):
    """
    Sphere potential, a nD potential energy surface with a spherical well.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def forward(self, positions):
        energies = torch.sum(positions ** 2, axis=-1, keepdim=True)
        forces = self.calculate_conservative_forces(energies, positions)
        return PotentialOutput(energies=energies, forces=forces)
