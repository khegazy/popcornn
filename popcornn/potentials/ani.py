import torch
from torch_geometric.nn import radius_graph
from torch.nn.functional import one_hot
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torchani.calculator import ANICalculator
from torchani.units import HARTREE_TO_EV

from .base_potential import BasePotential, PotentialOutput

class AniPotential(BasePotential):
    def __init__(self, model_path, **kwargs):
        """
        Constructor for NewtonNetPotential

        Parameters
        ----------
        model_path: str or list of str
            path to the model. eg. '5k/models/best_model_state.tar'
        settings_path: str or list of str
            path to the .yml setting path. eg. '5k/run_scripts/config_h2.yml'
        device: 
            device to run model. eg. 'cpu', ['cuda:0', 'cuda:1']
        kwargs
        """
        super().__init__(**kwargs)
        self.model = self.load_model(model_path)
        self.n_eval = 0

    
    def forward(self, points):
        data = self.data_formatter(points)
        pred = self.model(data)
        self.n_eval += 1
        energy = pred.energies.view(*points.shape[:-1], 1) * HARTREE_TO_EV
        return PotentialOutput(energy=energy)
        

    def load_model(self, model_path):
        calc = ANICalculator(model_path)
        model = calc.model
        model.eval()
        model.requires_grad_(False)
        return model
    
    def data_formatter(self, pos):
        n_atoms = self.n_atoms
        n_data = pos.numel() // (n_atoms * 3)
        z = self.numbers.repeat(n_data, 1)
        pos = pos.view(n_data, n_atoms, 3)
        return (z, pos)
