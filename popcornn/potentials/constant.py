import torch
from .base_potential import BasePotential, PotentialOutput

class Constant(BasePotential):
    """Flat potential — energy is ``scale`` everywhere, force is zero. Test fixture only."""

    def __init__(self, scale=1., **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def forward(self, positions):
        return PotentialOutput(
            energies=self.scale,
            forces=torch.zeros_like(self.positions)
        )