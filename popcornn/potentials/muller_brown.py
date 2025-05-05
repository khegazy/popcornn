import torch
from .base_potential import BasePotential, PotentialOutput


class MullerBrown(BasePotential):
    ai = [-200.0, -100.0, -170.0, 15.0]
    bi = [-1.0, -1.0, -6.5, 0.7]
    ci = [0.0, 0.0, 11.0, 0.6]
    di = [-10.0, -10.0, -6.5, 0.7]

    xi = [1.0, 0.0, -0.5, -1.0]
    yi = [0.0, 0.5, 1.5, 1.0]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def forward(self, positions):
        x, y = positions[:,0], positions[:,1]
        total = 0.0
        for i in range(4):
            b = self.bi[i]*(x - self.xi[i])*(x - self.xi[i])
            c = self.ci[i]*(x - self.xi[i])*(y - self.yi[i])
            d = self.di[i]*(y - self.yi[i])*(y - self.yi[i])
            total += self.ai[i]*torch.exp(b + c + d)

        return PotentialOutput(energies=total, forces=None)