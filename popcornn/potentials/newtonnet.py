
import torch
from torch_geometric.data import Data
from newtonnet.utils.ase_interface import MLAseCalculator
from newtonnet.data.neighbors import RadiusGraph
from newtonnet.models.output import NullAggregator

from .base_potential import BasePotential, PotentialOutput

class NewtonNetPotential(BasePotential):
    """
    Wrapper around NewtonNet — equivariant Newtonian-message-passing
    MLIP. Uses NewtonNet's ASE interface to load the checkpoint, then
    rewires the radius-graph transform so popcornn can call it
    directly with batched positions.
    """

    def __init__(self, model_path, **kwargs):
        """
        Parameters
        ----------
        model_path : str
            Path to a saved NewtonNet checkpoint.
        """
        super().__init__(**kwargs)
        self.model = self.load_model(model_path)
        self.transform = RadiusGraph(self.model.embedding_layer.norm.r)
        self.n_eval = 0


    def forward(self, positions):
        data = self.data_formatter(positions)
        pred = self.model(data.z, data.disp, data.edge_index, data.batch)
        self.n_eval += 1
        energies = pred.energy.unsqueeze(-1)
        forces = pred.gradient_force
        # energies_decomposed = energies_decomposed.view(-1, self.n_atoms)
        # return PotentialOutput(energies_decomposed=energies_decomposed)
        forces = forces.view(*positions.shape)
        return PotentialOutput(energies=energies, forces=forces)


    def load_model(self, model_path):
        calc = MLAseCalculator(model_path, properties=['energy', 'forces'], device=self.device)
        model = calc.models[0]
        # model.aggregators[0] = NullAggregator()
        model.eval()
        model.output_layers[1].create_graph = True
        model.to(self.dtype)
        model.requires_grad_(False)
        model.embedding_layer.requires_dr = False
        return model

    def data_formatter(self, pos):
        n_atoms = self.n_atoms
        n_data = pos.numel() // (n_atoms * 3)
        z = self.atomic_numbers.repeat(n_data)
        pos = pos.view(n_data * n_atoms, 3)
        cell = torch.zeros((n_data, 3, 3), device=self.device)
        batch = torch.arange(n_data, device=self.device).repeat_interleave(n_atoms)
        data = Data(pos=pos, z=z, cell=cell, batch=batch)
        
        return self.transform(data)