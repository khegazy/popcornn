import torch
from .base_potential import BasePotential, PotentialOutput

class Constant(BasePotential):
    def __init__(self, scale=1., **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def forward(self, positions):
        return PotentialOutput(
            energies=self.scale,
            forces=torch.zeros_like(self.positions)
        )