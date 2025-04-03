import torch
from torch_geometric.nn import radius_graph
from torch.nn.functional import one_hot
from torch_geometric.data import Batch
from torch_geometric.data import Data

from .base_potential import BasePotential, PotentialOutput

class LeftNetPotential(BasePotential):
    def __init__(self, model_path, use_autograd=True, **kwargs):
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
        self.use_autograd = use_autograd
        self.n_eval = 0
        self.one_hots = one_hot(self.numbers, num_classes=118)[:, [1, 6, 7, 8, 0]].double()

    
    def forward(self, points):
        data = self.data_formatter(points)
        if self.use_autograd:
            pred = self.model.forward_autograd(data)
        else:
            pred = self.model.forward(data)
        self.n_eval += 1
        energy = pred[0]
        force = pred[1]
        energy = energy.view(*points.shape[:-1], 1)
        # return PotentialOutput(energy=energy)
        force = force.view(*points.shape)
        return PotentialOutput(energy=energy, force=force)
        

    def load_model(self, model_path):
        model = torch.load(model_path, weights_only=False, map_location=self.device).double()
        model.eval()
        model.requires_grad_(False)
        return model
    
    def data_formatter(self, pos):
        n_atoms = torch.tensor(self.n_atoms, device=self.device)
        n_data = pos.numel() // (n_atoms * 3)

        data = Data(
            natoms=n_atoms.repeat(n_data),
            pos=pos.view(n_data * n_atoms, 3),
            one_hot=self.one_hots.repeat(n_data, 1),
            charges=self.numbers.repeat(n_data),
            batch=torch.arange(n_data, device=self.device).repeat_interleave(n_atoms),
            ae=n_atoms.repeat(n_data),
        )
        return data
