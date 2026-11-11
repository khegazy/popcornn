from __future__ import annotations

import torch

from .base_potential import BasePotential, PotentialOutput


class MullerBrown(BasePotential):
    """
    Muller-Brown potential, a 2D potential energy surface with three minima and
    two saddle points.


    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.A = torch.tensor([-200, -100, -170, 15], dtype=self.dtype, device=self.device)
        self.a = torch.tensor([-1, -1, -6.5, 0.7], dtype=self.dtype, device=self.device)
        self.b = torch.tensor([0, 0, 11, 0.6], dtype=self.dtype, device=self.device)
        self.c = torch.tensor([-10, -10, -6.5, 0.7], dtype=self.dtype, device=self.device)
        self.x0 = torch.tensor([1, 0, -0.5, -1], dtype=self.dtype, device=self.device)
        self.y0 = torch.tensor([0, 0.5, 1.5, 1], dtype=self.dtype, device=self.device)

    def forward(self, positions):
        x, y = positions[:, 0, None], positions[:, 1, None]
        energies = torch.sum(
            self.A * torch.exp(
                self.a * (x - self.x0) ** 2
                + self.b * (x - self.x0) * (y - self.y0)
                + self.c * (y - self.y0) ** 2
            ),
            axis=-1, keepdim=True
        )
        forces = self.calculate_conservative_forces(energies, positions)
        return PotentialOutput(energies=energies, forces=forces)
