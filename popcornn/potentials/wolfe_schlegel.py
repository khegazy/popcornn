
import torch

from .base_potential import BasePotential, PotentialOutput

class WolfeSchlegel(BasePotential):  # TODO: rename to Wolfe-Quapp potential
    """
    Wolfe–Quapp 2D analytic potential. Two minima, two saddles.

    Reference: Wolfe & Quapp, *J. Chem. Phys.* (2005),
    https://doi.org/10.1063/1.1885467 (note: the cross-term coefficient
    in the source code is 0.2 rather than the paper's 0.3 — a known
    typo, see the inline ``TODO``).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, positions):
        x = positions[:,0]
        y = positions[:,1]
        energies = 10*(x**4 + y**4 - 2*x**2 - 4*y**2\
            + x*y + 0.2*x + 0.1*y)  # TODO: it should be 0.3*x not 0.2*x, https://doi.org/10.1063/1.1885467
        energies = energies.unsqueeze(-1)
        forces = self.calculate_conservative_forces(energies, positions)
        return PotentialOutput(
            energies=energies,
            forces=forces
        )