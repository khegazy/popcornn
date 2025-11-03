import torch
from torch_geometric.utils import to_dense_batch
from ase.data import covalent_radii

from .base_potential import BasePotential, PotentialOutput
from popcornn.tools import radius_graph

class LennardJones(BasePotential):
    def __init__(self, epsilon=1.0, sigma=1.0, cutoff=None, **kwargs):
        """
        Constructor for the Lennard-Jones Potential.

        The potential is given by:
        E_ij = 4 * epsilon * ((sigma / r_ij) ** 12 - (sigma / r_ij) ** 6)
        E = sum_{i<j} E_ij

        r_ij is the distance between atoms i and j, under minimum image convention.
        """
        super().__init__(**kwargs)
        assert self.n_atoms is not None, "Number of atoms must be defined."
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff
    
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
        e = 4 * self.epsilon * ((self.sigma / r) ** 12 - (self.sigma / r) ** 6)
        if self.cutoff is not None:
            e -= 4 * self.epsilon * ((self.sigma / self.cutoff) ** 12 - (self.sigma / self.cutoff) ** 6)
        energies_decomposed, _ = to_dense_batch(e, batch=graph_dict['edge_index'][1] // n_atoms)
        energies = torch.sum(energies_decomposed, dim=-1, keepdim=True)

        f = (
            -24 * self.epsilon * (2 * (self.sigma / r) ** 12 - (self.sigma / r) ** 6) / r ** 2
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
            forces_decomposed=forces_decomposed
        )
