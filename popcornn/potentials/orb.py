import torch
from torch_geometric.nn import radius_graph
from torch.nn.functional import one_hot
from orb_models.forcefield.base import AtomGraphs

from .base_potential import BasePotential, PotentialOutput

class OrbPotential(BasePotential):
    def __init__(self, model_path, use_autograd=False, **kwargs):
        """
        Constructor for Orb Potential

        Parameters
        ----------
        model_path: str
            path to the model. eg. 'weights/orb/model.pt'
        """
        super().__init__(**kwargs)
        self.model = self.load_model(model_path)
        self.use_autograd = use_autograd
        self.n_eval = 0
        self.atomic_numbers_embedding = one_hot(self.atomic_numbers, num_classes=118).double()

    
    def forward(self, positions):
        data = self.data_formatter(positions)
        pred = self.model.predict(data)
        # pred = self.model(data.to_dict(), training=True)
        self.n_eval += 1
        if self.use_autograd:
            energies = pred['graph_pred'].view(*positions.shape[:-1], 1)
            return PotentialOutput(energies=energies)
        else:
            energies = pred['graph_pred'].view(*positions.shape[:-1], 1)
            forces = pred['node_pred'].view(*positions.shape)
            forces = forces.view(*positions.shape)
            return PotentialOutput(energies=energies, forces=forces)
        

    def load_model(self, model_path):
        model = torch.load(model_path, weights_only=False, map_location=self.device)
        model.to(torch.double)
        model.eval()
        model.requires_grad_(False)
        return model
    
    def data_formatter(self, pos):
        n_atoms = torch.tensor(self.n_atoms, device=self.device)
        n_data = pos.numel() // (n_atoms * 3)

        batch = torch.arange(n_data, device=self.device).repeat_interleave(n_atoms)
        positions = pos.view(n_data * n_atoms, 3)
        recievers, senders = radius_graph(positions, batch=batch, r=10.0, max_num_neighbors=20)
        n_node = n_atoms.repeat(n_data)
        n_edge = torch.tensor([len(recievers)], device=self.device)
        atomic_numbers = self.atomic_numbers.repeat(n_data)
        atomic_numbers_embedding = self.atomic_numbers_embedding.repeat(n_data, 1)
        node_features = {
            'atomic_numbers': atomic_numbers,
            'atomic_numbers_embedding': atomic_numbers_embedding,
            'positions': positions,
        }
        vectors = positions[recievers] - positions[senders]
        r = torch.norm(vectors, dim=-1)
        edge_features = {
            'vectors': vectors,
            'r': r,
        }
        cell = self.cell.repeat(n_data, 1, 1)
        system_features = {
            'cell': cell,
        }

        data = AtomGraphs(
            senders,
            recievers,
            n_node,
            n_edge,
            node_features,
            edge_features,
            system_features,
        )
        return data
