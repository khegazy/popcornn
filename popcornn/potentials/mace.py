import torch
from torch_geometric.nn import radius_graph
from torch.nn.functional import one_hot
from torch_geometric.data import Data

from .base_potential import BasePotential, PotentialOutput

class MacePotential(BasePotential):
    def __init__(self, model_path, **kwargs):
        """
        Constructor for MACE Potential

        Parameters
        ----------
        model_path: str
            path to the model. eg. 'weights/mace/model.pt'
        """
        super().__init__(**kwargs)
        self.model = self.load_model(model_path)
        self.n_eval = 0
        self.node_attrs = one_hot(self.atomic_numbers, num_classes=118)[:, self.model.atomic_numbers].double()

    
    def forward(self, positions):
        data = self.data_formatter(positions)
        pred = self.model(data.to_dict(), compute_force=False)
        # pred = self.model(data.to_dict(), training=True)
        self.n_eval += 1
        energies = pred['energy'].view(*positions.shape[:-1], 1)
        # forces = pred['forces'].view(*positions.shape)
        forces = self.calculate_conservative_forces(energies, positions)
        # return PotentialOutput(energies=energies)
        forces = forces.view(*positions.shape)
        return PotentialOutput(energies=energies, forces=forces)
        

    def load_model(self, model_path):
        # model = mace_off(device=self.device).models[0]
        # state_dict = torch.load(model_path, map_location=self.device)
        # model.load_state_dict(state_dict)
        model = torch.load(model_path, weights_only=False, map_location=self.device)
        model.eval()
        model.requires_grad_(False)
        return model
    
    def data_formatter(self, pos):
        n_atoms = self.n_atoms
        n_data = pos.numel() // (n_atoms * 3)
        
        positions = pos.view(n_data * n_atoms, 3)
        cell = self.cell.repeat(n_data, 1)
        node_attrs = self.node_attrs.repeat(n_data, 1)
        batch = torch.arange(n_data, device=self.device).repeat_interleave(n_atoms)
        ptr = torch.arange(0, n_data + 1, device=self.device) * n_atoms
        edge_index = radius_graph(positions, r=self.model.r_max, batch=batch)
        shifts = torch.zeros(edge_index.shape[1], 3, device=self.device)
        unit_shifts = torch.zeros(edge_index.shape[1], 3, device=self.device)
        data = Data(
            positions=positions,
            cell=cell,
            node_attrs=node_attrs,
            edge_index=edge_index,
            shifts=shifts,
            unit_shifts=unit_shifts,
            batch=batch,
            ptr=ptr,
        )
        return data
