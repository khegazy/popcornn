import torch
from torch_geometric.utils import to_dense_batch
from ase.data import covalent_radii

from .base_potential import BasePotential, PotentialOutput
from popcornn.tools import radius_graph

class RepelPotential(BasePotential):
    """
    Soft-repulsive potential for geodesic interpolation.

    From Zhu, Thompson & Martínez, *J. Chem. Phys.* **150**, 164103
    (2019). Energy per pair:

    .. math::
        E_{ij} = e^{-\\alpha (r_{ij} - r_0)/r_0} + \\beta \\, r_0 / r_{ij}

    where ``r_0`` is the sum of covalent radii. Use this as a first
    optimization stage with ``path_integrand_names: geodesic`` to fix atom
    clashes before handing the path off to an MLIP.
    """

    def __init__(
            self,
            alpha=1.7,
            beta=0.01,
            cutoff=None,
            **kwargs,
        ):
        """
        Parameters
        ----------
        alpha : float, default=1.7
            Exponential decay rate.
        beta : float, default=0.01
            Inverse-distance term weight.
        cutoff : float, optional
            Cutoff distance. ``None`` keeps all pairs.
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.cutoff = cutoff
        self.radii = torch.tensor([covalent_radii[n] for n in self.atomic_numbers], device=self.device, dtype=self.dtype)
    
    def forward(self, positions):
        positions_3d = positions.view(-1, self.n_atoms, 3)
        n_data, n_atoms, _ = positions_3d.shape
        graph_dict = radius_graph(
            positions=positions_3d,
            cell=self.cell,
            pbc=self.pbc,
            cutoff=self.cutoff,
            max_neighbors=-1,
        )
        r = graph_dict['edge_distance']
        v = graph_dict['edge_distance_vec']
        r0 = self.radii[graph_dict['edge_index'] % n_atoms].sum(dim=0)  # sum of covalent radii for each edge
        e = torch.exp(-self.alpha * (r - r0) / r0) + self.beta * r0 / r
        if self.cutoff is not None:
            e -= torch.exp(-self.alpha * (self.cutoff - r0) / r0) + self.beta * r0 / self.cutoff
        energies_decomposed, _ = to_dense_batch(e, batch=graph_dict['edge_index'][1] // n_atoms)
        energies = torch.sum(energies_decomposed, dim=-1, keepdim=True)
        
        f = (
            - torch.exp(-self.alpha * (r - r0) / r0) * self.alpha / r0 / r - self.beta * r0 / r ** 3
        ).unsqueeze(-1) * v
        forces_decomposed = torch.zeros(len(f), n_atoms, 3, device=self.device, dtype=self.dtype)
        forces_decomposed[torch.arange(len(f), device=self.device), graph_dict['edge_index'][0] % n_atoms] = -f
        forces_decomposed[torch.arange(len(f), device=self.device), graph_dict['edge_index'][1] % n_atoms] = f
        forces_decomposed, _ = to_dense_batch(forces_decomposed, batch=graph_dict['edge_index'][1] // n_atoms)
        forces_decomposed = forces_decomposed.view(*forces_decomposed.shape[:-2], -1)
        forces = torch.sum(forces_decomposed, dim=-2, keepdim=False)

        return PotentialOutput(
            energies=energies,
            energies_decomposed=energies_decomposed,
            forces=forces,
            forces_decomposed=forces_decomposed,
        )

