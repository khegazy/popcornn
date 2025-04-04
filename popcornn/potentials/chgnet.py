import torch
from torch_geometric.nn import radius_graph
from torch.nn.functional import one_hot
from torch_geometric.data import Data

from .base_potential import BasePotential, PotentialOutput

class CHGNetPotential(BasePotential):
    def __init__(self, model_path, **kwargs):
        """
        Constructor for CHGNet Potential

        Parameters
        ----------
        model_path: str
            path to the model. eg. 'weights/chg/model.pt'
        """
        raise NotImplementedError("CHGNetPotential is not implemented yet.")
        super().__init__(**kwargs)
        self.model = self.load_model(model_path)
        self.n_eval = 0
        self.node_attrs = one_hot(self.numbers, num_classes=118)[:, self.model.atomic_numbers].double()

    
    def forward(self, points):
        data = self.data_formatter(points)
        pred = self.model(data.to_dict(), compute_force=False)
        self.n_eval += 1
        energy = pred['energy'].view(*points.shape[:-1], 1)
        # force = pred['forces'].view(*points.shape)
        return PotentialOutput(energy=energy)
        # force = force.view(*points.shape)
        # return PotentialOutput(energy=energy, force=force)
        

    def load_model(self, model_path):
        model = mace_off(device=self.device).models[0]
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
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
